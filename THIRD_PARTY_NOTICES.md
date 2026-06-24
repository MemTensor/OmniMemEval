# Third-Party Notices

MemEval code is licensed under the repository [LICENSE](./LICENSE). Benchmark
datasets keep their upstream licenses. The MemEval code license does not
relicense external datasets.

## LoCoMo

- Source: https://github.com/snap-research/locomo
- Paper: https://arxiv.org/abs/2402.17753
- Data file used by MemEval: `data/locomo/locomo10.json`
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0)
- Notes: the data file is not committed to this repository. Run
  `python data/locomo/prepare_locomo.py` to download it from upstream. Use of
  the data must remain non-commercial and include attribution to the LoCoMo
  authors.

## LongMemEval

- Source repository: https://github.com/xiaowu0162/LongMemEval
- Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- Paper: https://arxiv.org/abs/2410.10813
- Data file used by default: `data/longmemeval/longmemeval_s_cleaned.json`
- License: MIT
- Notes: LongMemEval data is not committed to this repository. Run
  `python data/longmemeval/prepare_longmemeval.py` to download the default S
  variant.
