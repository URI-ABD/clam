# A TODO list for the next paper

## Reviews from ISMB 2026

Ignore the first review, which is from a troll.

### Datasets

- [ ] [BAliBASE](https://www.re3data.org/repository/r3d100012946)
- [ ] [HOMSTRAD](https://pmc.ncbi.nlm.nih.gov/articles/PMC2143859/)

### MSA Algorithms

Several are available in [BioPython](https://biopython.org/docs/1.75/api/Bio.Align.Applications.html). Use the following, and any others that might be relevant:

- [ ] [Clustal Omega]
- [ ] [MAFFT]
- [ ] [MUSCLE]
- [ ] [T-Coffee]

### Pairwise Alignment Algorithms

- [ ] [Wavefront Alignment](https://academic.oup.com/bioinformatics/article/37/4/456/5904262) and [source code](https://github.com/leomrtns/WFA)

### Reviewer 4

The only good and constructive review.

## Algorithmic Improvements and Benchmarks

- [x] Change the `Sequence` trait to store gaps in the negative space between ranges, where the ranges represent contiguous sequences of non-gap characters from the original sequence. This will save on memory and allow for scaling to 10M+ sequences without running out of memory.
- [ ] Benchmark the various partition strategies and their impact on the final MSA quality.
- [ ] Rework the alignment quality measures to have their names and definitions be more aligned with the standard literature on MSA evaluation metrics.

## Reproducibility

- [ ] Docker container with pre-compiled code and all dependencies for running the benchmarks and evaluations.
