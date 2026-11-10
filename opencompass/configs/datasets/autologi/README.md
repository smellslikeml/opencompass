# AutoLogi

[AutoLogi](https://arxiv.org/abs/2502.16906) is an open-ended,
program-verified logic-puzzle reasoning benchmark. Each puzzle is a
constraint-satisfaction problem (scheduling, ordering, selection, ...)
whose prompt asks the model for a complete assignment, and whose
correctness is decided by executing the puzzle's packaged
`inputs_check` and constraint predicates against the extracted answer.

This configuration evaluates both language editions — `autologi-en`
(1,575 puzzles) and `autologi-cn` (883 puzzles). The data is loaded at
runtime from the Apache-2.0 source dataset
[`qzhu/AutoLogi`](https://huggingface.co/datasets/qzhu/AutoLogi) and is
not redistributed with OpenCompass. The language split follows the data
files (`AutoLogi_en.jsonl` / `AutoLogi_cn.jsonl`), not the per-record
`lang` field, which is mislabeled in the source data.

Grading is pass@1 accuracy: the answer is extracted from the model
generation (last fenced code block, else the last balanced `{...}` /
`[...]` literal) and parsed as data with `ast.literal_eval` — it is
never executed. Only the dataset-packaged verifier code runs, inside a
forked subprocess with a timeout. Extraction failures and timeouts
count as incorrect.

To use the dataset, import `autologi_datasets` in an evaluation config
and pair it with the model configuration you want to evaluate. Cite the
paper when reporting results.
