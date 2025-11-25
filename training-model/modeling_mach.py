import torch
from torch import nn
from torch.nn import functional as F

from .configuration_mach import StripedHyenaConfig
# from configuration_mach import StripedHyenaConfig


from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput, CausalLMOutput, CausalLMOutputWithPast, SequenceClassifierOutput
from transformers.utils import logging
from typing import Optional, Tuple, Union, List
from stripedhyena.model import StripedHyena
from stripedhyena.utils import dotdict
from stripedhyena.cache import InferenceParams
from stripedhyena.engine import HyenaInferenceEngine
from stripedhyena.layers import RMSNorm, VocabParallelEmbedding
from stripedhyena.utils import dotdict, column_split

from dataclasses import dataclass

import wandb
from wandb import AlertLevel

logger = logging.get_logger(__name__)


class StripedHyenaPreTrainedModel(PreTrainedModel):
    config_class = StripedHyenaConfig
    base_model_prefix = "sh"
    supports_gradient_checkpointing = False
    _no_split_modules = ["AttentionBlock", "ParallelGatedConvBlock"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_missing = [r"freq"]
    _keys_to_ignore_on_load_unexpected = [r"fftconv", r"twiddle_factors"]
    _supports_flash_attn_2 = True

class StripedHyenaModelForCausalLM(StripedHyenaPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        model_config = dotdict(config.to_dict())
        self.backbone = StripedHyena(model_config)
        self.backbone.gradient_checkpointing = False
        self.config = config
        vocab_size = config.vocab_size
        if vocab_size % config.make_vocab_size_divisible_by != 0:
            vocab_size += config.make_vocab_size_divisible_by - (
                vocab_size % config.make_vocab_size_divisible_by
            )
        self.vocab_size = vocab_size
        self.post_init()
        self.force_dtype()

    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues() 
        
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.backbone.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache:
            if self.backbone.gradient_checkpointing and self.backbone.training:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
            elif labels is not None:
                logger.warning_once(
                    "`use_cache=True` is incompatible with loss calculation. Setting `use_cache=False`..."
                )
                use_cache = False

        inputs = input_ids
        if use_cache:
            if past_key_values is None:
                past_key_values = self.backbone.initialize_inference_params()

                batch_size = input_ids.shape[0]
                past_key_values["mha"].max_batch_size = batch_size
                past_key_values["hyena"].max_batch_size = batch_size
            else:
                seqlen_offset = past_key_values["mha"].seqlen_offset
                if seqlen_offset == 0:
                    # second loop through generate will have prompt_len + 1 as seqlen
                    seqlen_offset = input_ids.shape[-1] - 1
                    past_key_values["hyena"].seqlen_offset = seqlen_offset
                    past_key_values["mha"].seqlen_offset = seqlen_offset
                else:
                    past_key_values["mha"].seqlen_offset += 1
                    past_key_values["hyena"].seqlen_offset += 1

                inputs = input_ids[
                    :,
                    -1:,
                ]

        logits, past_key_values = self.backbone(
            inputs,
            padding_mask=attention_mask,
            inference_params_dict=past_key_values if use_cache else None,
        )
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)

            if (loss == 0) | (loss.isnan()):
                wandb.alert(
                    title="Invalid loss value",
                    text="Loss is zero or NaN",
                    level=AlertLevel.WARN,
                    # wait_duration=600
                )
                wandb.finish(exit_code=600)

        if return_dict:
            return CausalLMOutputWithPast(
                logits=logits,
                hidden_states=None,
                past_key_values=past_key_values if use_cache else None,
                loss=loss,
            )
        else:
            return logits

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, past_key_values=None, **kwargs
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }


class StripedHyenaForEmbeddings(StripedHyena):
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.pas_token_id = tokenizer.convert_tokens_to_ids(['P'])[0]

    def forward(self, x, inference_params_dict=None, padding_mask=None):
        B, L = x.shape  # B: batch size, L: sequence length
        self.pas_token_indices = (x == self.pas_token_id).nonzero(as_tuple=True)
        x = self.embedding_layer.embed(x)  # Shape: (B, L, 128)

        if padding_mask is not None:
            mask_sum = padding_mask.sum(dim=1, keepdim=True)
            layer_avg_emb = (x * padding_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        else:
            layer_avg_emb = x.mean(dim=1)

        pas_token_emb = x[self.pas_token_indices]
        
        layer_avg_embeds = [layer_avg_emb]
        pas_token_embeds = [pas_token_emb]

        if inference_params_dict is not None:
            x, block_layer_avg_embeds, block_pas_token_embeds = self.stateful_forward(x, inference_params_dict, padding_mask)
        else:    
            x, block_layer_avg_embeds, block_pas_token_embeds = self.stateless_forward(x, padding_mask)

        layer_avg_embeds.extend(block_layer_avg_embeds)
        pas_token_embeds.extend(block_pas_token_embeds)

        x = self.norm(x) if self.norm else x
        x = self.unembed.unembed(x)

        layer_avg_embeds = torch.stack(layer_avg_embeds, dim=1)  # Shape: (B, 17, 128)
        pas_token_embeds = torch.stack(pas_token_embeds, dim=1)  # Shape: (B, 17, 128)
        
        return x, layer_avg_embeds, pas_token_embeds

    def stateful_forward(self, x, inference_params_dict, padding_mask=None):
        layer_avg_embeds = []
        pas_token_embeds = []
        
        for block_idx, block in enumerate(self.blocks):
            block_name = "mha" if block_idx in self.config.attn_layer_idxs else "hyena"
            inference_params = inference_params_dict[block_name]
            x, _ = block(x, inference_params=inference_params)

            if padding_mask is not None:
                mask_sum = padding_mask.sum(dim=1, keepdim=True)
                layer_avg_emb = (x * padding_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
            else:
                layer_avg_emb = x.mean(dim=1)
                
            pas_token_emb = x[self.pas_token_indices]
            
            layer_avg_embeds.append(layer_avg_emb)
            pas_token_embeds.append(pas_token_emb)

        return x, layer_avg_embeds, pas_token_embeds

    def stateless_forward(self, x, padding_mask=None):
        layer_avg_embeds = []
        pas_token_embeds = []

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)

        for _, block in enumerate(self.blocks):
            x, _ = block(x, inference_params=None, padding_mask=padding_mask)

            if padding_mask is not None:
                mask_sum = padding_mask.sum(dim=1, keepdim=True)
                layer_avg_emb = (x * padding_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
            else:
                layer_avg_emb = x.mean(dim=1)
                
            pas_token_emb = x[self.pas_token_indices]
            
            layer_avg_embeds.append(layer_avg_emb)
            pas_token_embeds.append(pas_token_emb)

        return x, layer_avg_embeds, pas_token_embeds

@dataclass
class CausalLMEmbeddingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    layer_avg_embeds: torch.FloatTensor = None
    pas_token_embeds: torch.FloatTensor = None

class StripedHyenaModelForExtractingEmbeddings(StripedHyenaPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config, tokenizer, **kwargs):
        super().__init__(config, **kwargs)
        self.tokenizer = tokenizer
        model_config = dotdict(config.to_dict())
        self.backbone = StripedHyenaForEmbeddings(model_config, self.tokenizer)
        self.backbone.gradient_checkpointing = False
        self.config = config
        self.post_init()
        self.force_dtype()

    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues() 
        
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.backbone.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache:
            if self.backbone.gradient_checkpointing and self.backbone.training:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
            elif labels is not None:
                logger.warning_once(
                    "`use_cache=True` is incompatible with loss calculation. Setting `use_cache=False`..."
                )
                use_cache = False

        inputs = input_ids
        if use_cache:
            if past_key_values is None:
                past_key_values = self.backbone.initialize_inference_params()

                batch_size = input_ids.shape[0]
                past_key_values["mha"].max_batch_size = batch_size
                past_key_values["hyena"].max_batch_size = batch_size
            else:
                seqlen_offset = past_key_values["mha"].seqlen_offset
                if seqlen_offset == 0:
                    # second loop through generate will have prompt_len + 1 as seqlen
                    seqlen_offset = input_ids.shape[-1] - 1
                    past_key_values["hyena"].seqlen_offset = seqlen_offset
                    past_key_values["mha"].seqlen_offset = seqlen_offset
                else:
                    past_key_values["mha"].seqlen_offset += 1
                    past_key_values["hyena"].seqlen_offset += 1

                inputs = input_ids[
                    :,
                    -1:,
                ]

        _, layer_avg_embeds, pas_token_embeds = self.backbone(
            inputs,
            padding_mask=attention_mask,
            inference_params_dict=past_key_values if use_cache else None,
        )

        loss = torch.tensor(0.0)
        
        return CausalLMEmbeddingOutput(
            loss=loss,
            layer_avg_embeds=layer_avg_embeds,
            pas_token_embeds=pas_token_embeds)


class StripedHyenaWithHiddenLayerOutput(StripedHyena):
    def __init__(self, config):
        super().__init__(config)
        # self.main_input_name = 'input_ids_1'

    def forward(self, x, inference_params_dict=None, padding_mask=None):
        L = x.shape[1]
        x = self.embedding_layer.embed(x)
        if inference_params_dict is not None:
            x, inference_params_dict_out = self.stateful_forward(
                x,
                inference_params_dict=inference_params_dict,
            )
        else:
            x, inference_params_dict_out = self.stateless_forward(x, padding_mask=padding_mask)

        x = self.norm(x)
        # x = self.unembed.unembed(x)
        return x, inference_params_dict_out        

class StripedHyenaModelForPairSequenceClassification(StripedHyenaPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        model_config = dotdict(config.to_dict())
        self.backbone = StripedHyenaWithHiddenLayerOutput(model_config)
        self.backbone.gradient_checkpointing = False
        self.config = config

        self.main_input_name = 'input_ids_1'
        self.cls = nn.Linear(config.hidden_size * 2, 2, bias=False)

        self.post_init()
        self.force_dtype()
    
    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues() 
        
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.backbone.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def forward(
        self,
        input_ids_1: torch.LongTensor = None,
        attention_mask_1: Optional[torch.LongTensor] = None,
        input_ids_2: torch.LongTensor = None,
        attention_mask_2: Optional[torch.LongTensor] = None,
        seq_len: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits_1, _ = self.backbone(
            input_ids_1,
            padding_mask=attention_mask_1,
            inference_params_dict=None,
        )

        logits_2, _ = self.backbone(
            input_ids_2,
            padding_mask=attention_mask_2,
            inference_params_dict=None,
        )

        num_seqs = seq_len.shape[0]
        index = seq_len.view(num_seqs, 1, 1).expand(num_seqs, 1, 128)
        last_token_logits_1 = torch.gather(logits_1, 1, index).squeeze(1) ## first token is [CLS], last token is [SEP], so we need to get the second last token [E]
        last_token_logits_2 = torch.gather(logits_2, 1, index).squeeze(1)

        last_tokens_logits = torch.cat([last_token_logits_1, last_token_logits_2], dim=1)
        logits = self.cls(last_tokens_logits)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))

            if (loss == 0) | (loss.isnan()):
                wandb.alert(
                    title="Invalid loss value",
                    text="Loss is zero or NaN",
                    level=AlertLevel.WARN,
                    wait_duration=600
                )

        if return_dict:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=None,
                attentions=None
            )
        else:
            return logits
