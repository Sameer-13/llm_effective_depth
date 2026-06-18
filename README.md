# Codebase for our paper "Do Language Models Use Their Depth Efficiently?"

We have two independent codebases, one for the analysis, which is most of the paper, and one for traning and finetuning. Each of the subfolders has its own readme file. Please check those for more details.

```bash
export NDIF_TOKEN=""
```
Normal run with logs:
```bash
python analyze_future_effects.py qwen3_8b --parts layer --language arabic --data-path /home/sabeasm/llm_effective_depth/data/single_arabic_case.json --output-dir output/arabic/single/full_comp > arabic_run.log 2>&1 &
```
Background run with logs:
```bash
nohup python analyze_future_effects.py qwen3_8b --parts layer --language arabic --data-path /home/sabeasm/llm_effective_depth/data/single_arabic_case.json --output-dir output/arabic/single/full_comp > arabic_run.log 2>&1 &
```