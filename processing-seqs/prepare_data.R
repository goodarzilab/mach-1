library(optparse)
library(data.table)
library(stringr)
library(rtracklayer)

# Development
# -----------
# gtf_file = "~/databases/data/GENCODE/gencode.v44.annotation.gtf.gz"
# gtf_file = "~/repositories/lorna-sh/example-CD44.gtf.gz"


option_list <- list(
  make_option(c("-s", "--species"), type = "character", default = 'human', help = "Species name (human or mouse)"),
  make_option(c("-f", "--flank_len"), type = "integer", default = 16, help = "Flanking length"),
  make_option(c("-m", "--max_seq_len"), type = "integer", default = 2^16, help = "Maximum sequence length"),
  make_option(c("--gtf_file"), type = "integer", default = 2^16, help = "Maximum sequence length"),
  make_option(c("--output_file"), type = "integer", default = 2^16, help = "Maximum sequence length")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

species <- opt$species
flank_len <- opt$flank_len
max_seq_len <- opt$max_seq_len
gtf_file <- opt$gtf_file
output_file <- opt$output_file

if (is.null(species)) {
  stop("Species must be provided using -s or --species option")
}

if (species == 'human') {
  
  library(BSgenome.Hsapiens.UCSC.hg38)
  ref_genome <- BSgenome.Hsapiens.UCSC.hg38
  species_token <- 'H'
  # gtf_file <- '/scratch/asabe/projects/foundation-model/preprocess/pre-mrna/data/databank_human_bambu_se_discovery.gtf'
  # assembly_report_file <- '/scratch/asabe/projects/pacbio/data/references/genome/GRCh38_latest_assembly_report.txt'
  # output_file <- '/scratch/asabe/projects/foundation-model/preprocess/pre-mrna/data/databank_human_bambu_se_discovery.preprocessed.updated.csv.gz'
  
} else if (species == 'mouse') {
  
  library(BSgenome.Mmusculus.UCSC.mm39)
  ref_genome <- BSgenome.Mmusculus.UCSC.mm39
  gtf_file <- '/scratch/asabe/projects/foundation-model/preprocess/pre-mrna/data/databank_mouse_bambu_se_discovery.gtf'
  assembly_report_file <- '/scratch/asabe/projects/pacbio/data/references/genome/GCF_000001635.27_GRCm39_assembly_report.txt'
  species_token <- 'M'
  output_file <- '/scratch/asabe/projects/foundation-model/preprocess/pre-mrna/data/databank_mouse_bambu_se_discovery.preprocessed.updated.csv.gz'
  
}

gtf <- import(gtf_file)
gtf <- as.data.table(gtf)

columns_to_keep <- c('seqnames', 'type', 'feature', 'start', 'end', 'width', 'strand', 'gene_id', 'transcript_id', 'exon_number')
columns_to_keep <- intersect(columns_to_keep, colnames(gtf))
gtf <- gtf[, columns_to_keep, with = F]
gtf <- gtf[type %in% c('transcript', 'exon')]
gtf[, exon_number := as.integer(exon_number)]
# gtf[transcript_id %like% 'Bambu', transcript_id := str_c(species_token, transcript_id)]
# gtf[gene_id %like% 'Bambu', gene_id := str_c(species_token, gene_id)]


# assembly_info <- fread(assembly_report_file,
#                       skip = '# Sequence-Name',
#                       header = T,
#                       select = c('GenBank-Accn', 'RefSeq-Accn', 'UCSC-style-name', 'Sequence-Length'),
#                       col.names = c('genbank_seqnames', 'refseq_seqnames', 'chr', 'chr_len'))

assembly_info = seqlengths(ref_genome)
assembly_info = data.frame(
    chr = names(assembly_info),
    chr_len = assembly_info
)

# setnames(assembly_info, 'genbank_seqnames', 'seqnames')
# gtf <- merge(gtf, assembly_info[, .(seqnames, chr)], by = 'seqnames', all.x = TRUE)
# gtf[!is.na(chr), seqnames := chr]
# gtf[, chr := NULL]

gtf <- merge(gtf, assembly_info, by.x = 'seqnames', by.y = 'chr', all.x = TRUE)
  
### First primary chromosomes, then random, then unlocalized
chr_order <- gtf[, .(seqnames)]
chr_order[, num := seqnames %>% str_extract('^chr([0-9]+|X|Y|M|Un)') %>% str_remove('chr')]

chr_order[, type := fcase(
  str_detect(seqnames, 'random'), 'random',
  str_detect(seqnames, 'alt'), 'alt',
  str_detect(seqnames, 'fix'), 'fix',
  str_detect(seqnames, 'chrUn'), 'unlocalized',
  default = 'primary'
)]

chr_order[, num := factor(num, levels = c(seq(22), 'X', 'Y', 'M', 'Un'), ordered = TRUE)]
chr_order[, type := factor(type, levels = c('primary', 'fix', 'random', 'unlocalized', 'alt'), ordered = TRUE)]
chr_order <- unique(chr_order)
setorder(chr_order, type, num)

gtf_chrs <- chr_order[, seqnames %>% unique()]
gtf[, seqnames := factor(seqnames, levels = gtf_chrs, ordered = TRUE)]
gtf <- gtf[order(seqnames)]


### Removing transcripts with 1 exon only
gtf[, num_exons := max(exon_number, na.rm = T), transcript_id]

gtf <- gtf[num_exons > 1]

gtf[, gene_start := min(start), gene_id]
gtf[, gene_end := max(end), gene_id]

### Adding flanking regions to the gene coords
gtf[, gene_start := gene_start - flank_len]
gtf[, gene_end := gene_end + flank_len]

gtf <- gtf[(gene_start >= 1) & (gene_end <= chr_len)]

gtf[, tokenized_gene_len := (gene_end - gene_start + 1) + 1 + 1 + 1 + 1 + 1] ## CLS, H/M, S, E, SEP 
gtf <- gtf[tokenized_gene_len <= max_seq_len]
gtf[, tokenized_gene_len := NULL]

### Reversing exon_number on the negative strand
gtf[strand == '-', exon_number := (num_exons - exon_number + 1)]

introns <- gtf[type == 'exon']
introns <- introns[, start_ := start]

introns[order(end), start := c(end[-.N] + 1, NA), transcript_id]
introns[order(end), end := c(start_[-1] - 1, NA), transcript_id]
introns[, start_ := NULL]
introns[, width := end - start + 1]
introns[, type := 'intron']
introns <- introns[!is.na(start)]

gtf <- rbind(gtf, introns)

non_rna <- gtf[type == 'exon']
non_rna[, first_exon_num := min(exon_number), transcript_id]
non_rna[, last_exon_num := max(exon_number), transcript_id]
non_rna[, is_first_exon := fifelse(exon_number == first_exon_num, T, F)]
non_rna[, is_last_exon := fifelse(exon_number == last_exon_num, T, F)]

non_rna <- non_rna[is_first_exon | is_last_exon]

non_rna[is_first_exon == T, end := start-1]
non_rna[is_first_exon == T, start := gene_start]

non_rna[is_last_exon == T, start := end+1]
non_rna[is_last_exon == T, end := gene_end]

non_rna[, width := end - start + 1]

# non_rna[, type := 'non_rna']
non_rna[is_first_exon == T, type := 'non_rna_upstream']
non_rna[is_last_exon == T, type := 'non_rna_downstream']

non_rna[, c('first_exon_num', 'last_exon_num', 'is_first_exon', 'is_last_exon') := NULL]

gtf <- rbind(gtf, non_rna)

gtf <- gtf[strand != '*']

invalid_trs <- gtf[(start <= 0) | (end <= 0) | (start > end), transcript_id]
print(str_glue('{length(invalid_trs)} invalid transcripts (due to negative coordinates)'))
gtf <- gtf[!(transcript_id %in% invalid_trs)]

## if the strand is '-', it'll revComp. the sequence, which we don't need at this step.
gtf[type != 'transcript', seq := getSeq(ref_genome,
                                        names = seqnames,
                                        start = start,
                                        end = end,
                                        as.character = TRUE)]
                                        # strand = strand
                    

invalid_trs <- gtf[str_count(seq, 'N') > 0, transcript_id %>% unique()]
print(str_glue('{length(invalid_trs)} invalid transcripts (due to N)'))
gtf <- gtf[!(transcript_id %in% invalid_trs)]

non_rna_transformer <- c('A' = 'W', 'C' = 'X', 'G' = 'Y', 'T' = 'Z')
transform_to_non_rna <- function(seq) {
  non_rna_transformer[
    seq %>% str_split('') %>% unlist()
  ] %>% str_c(collapse = '')
}

transform_to_non_rna_v <- Vectorize(transform_to_non_rna, 'seq')
gtf[type %like% 'non_rna', seq := transform_to_non_rna_v(seq)]

gtf[type == 'non_rna_upstream',  seq := str_c(seq, 'S')] ## Adding seq, S (TSS), will add the H/M token later
gtf[type == 'non_rna_downstream', seq := str_c('E', seq)] ## Adding E (TTS), flank_end
gtf[type == 'exon', seq := str_to_lower(seq)]

gtf[, type := factor(type, levels = c('non_rna_upstream', 'exon', 'intron', 'non_rna_downstream', 'transcript'), ordered = TRUE)]

tr_seqs <- gtf[type != 'transcript'][
  order(transcript_id, exon_number, type),
  .(seq = str_c(seq, collapse = '')),
  by = .(transcript_id, strand)]

complement_transformer <- c('A' = 'T', 'T' = 'A', 'C' = 'G', 'G' = 'C',
                           'a' = 't', 't' = 'a', 'c' = 'g', 'g' = 'c',
                           'W' = 'Z', 'Z' = 'W', 'X' = 'Y', 'Y' = 'X',
                           'D' = 'R', 'R' = 'D',
                           'S' = 'E', 'E' = 'S')

transform_to_reverse_complement <- function(seq) {
  complement_transformer[
    seq %>% str_split('') %>% unlist() %>% rev()
  ] %>% str_c(collapse = '')
}

transform_to_reverse_complement_v <- Vectorize(transform_to_reverse_complement, 'seq')

tr_seqs[strand == '-', seq := transform_to_reverse_complement_v(seq)]
tr_seqs[, seq := str_c(species_token, seq)]

tr_seqs[, seq_len := nchar(seq)]
tr_seqs[, strand := NULL]

tr_info <- gtf[type == 'transcript',
              c('seqnames', 'start', 'end', 'width', 'strand', 'gene_id', 'transcript_id', 'num_exons', 'gene_end', 'gene_start')] 

colnames(tr_info) <- c('chr', 'tr_start', 'tr_end', 'tr_len', 'strand', 'gene_id', 'transcript_id', 'num_exons', 'gene_end', 'gene_start')

tr_seqs <- merge(tr_seqs, tr_info, by = 'transcript_id', all.x = TRUE)

tr_seqs[, species := species]

setcolorder(tr_seqs, c('species', 'chr', 'strand', 'gene_start', 'gene_end', 'gene_id', 'tr_start', 'tr_end', 'transcript_id', 'num_exons', 'tr_len', 'seq_len', 'seq'))

setorderv(tr_seqs, c('species', 'chr', 'gene_start', 'gene_id', 'tr_start', 'transcript_id'))

include_header <- ifelse(species %like% 'human', TRUE, FALSE)
fwrite(tr_seqs, file = output_file, col.names = include_header)