# HotpotQA smoke data

`hotpotqa_dev_first2.json` contains the first two rows of the **distractor / validation**
split from [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa).
They are real benchmark questions, selected by row position, not outcome or difficulty.
Each retains all ten context documents in their original order. `source.json`
records IDs, retrieval URL, exact source-response hash and redistributed-file hash.
No source revision is claimed for the dataset viewer's unversioned rows API.

This subset is adapted from **HotpotQA: A Dataset for Diverse, Explainable Multi-hop
Question Answering**, Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W.
Cohen, Ruslan Salakhutdinov and Christopher D. Manning, EMNLP 2018.
[Dataset homepage](https://hotpotqa.github.io/) · [Paper](https://arxiv.org/abs/1809.09600).

The dataset and this adapted subset are licensed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
Changes: select rows 0–1, convert Hugging Face's columnar context to native
`[title, sentences]` pairs, rename `id` to `_id`, and omit type/level/support labels.
Questions, answers and context text are unchanged. This license applies to the
dataset subset, not automatically to repository software.

The launcher converts this file into separate public tasks and private grading
references. It never sends the `answer` field to the actor or memory helper.
Bundling the subset avoids downloads/DNS dependencies on Vista compute nodes.
Two rows are for integration smoke testing, not a reported HotpotQA score.
