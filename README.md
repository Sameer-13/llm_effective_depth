# Codebase for our paper "Do Language Models Use Their Depth Efficiently?"

We have two independent codebases, one for the analysis, which is most of the paper, and one for traning and finetuning. Each of the subfolders has its own readme file. Please check those for more details.

```bash
export NDIF_TOKEN=""
```
Normal run with logs:
```bash
python analyze_future_effects.py llama_3_1_8b_instruct --parts layer --language arabic --data-path /home/ubuntu/llm_effective_depth/data/single_arabic_case.json --output-dir output/llama_3_1_8b_instruct/arabic/single/full_comp > arabic_run.log 2>&1 &
```
Background run with logs:
```bash
nohup python analyze_future_effects.py qwen3_8b --parts layer --language arabic --data-path /home/ubuntu/llm_effective_depth/data/single_arabic_case.json --output-dir output/arabic/single/full_comp > arabic_run.log 2>&1 &
```


Command per model:
```bash
python analyze_future_effects.py llama_3_1_8b_instruct --parts layer --no-completions --no-accuracy --language arabic --data-path /home/ubuntu/llm_effective_depth/data/arabic_analysis_cases.json --output-dir output/llama_3_1_8b_instruct/arabic/mutli/layer_effect/

python analyze_future_effects.py llama_3_1_8b_instruct --parts layer --no-completions --no-accuracy --language english --data-path /home/ubuntu/llm_effective_depth/data/english_analysis_cases.json --output-dir output/llama_3_1_8b_instruct/english/mutli/layer_effect/

python logitlens.py llama_3_1_8b_instruct --language arabic --data-path /home/ubuntu/llm_effective_depth/data/arabic_analysis_cases.json --output-dir output/llama_3_1_8b_instruct/arabic/mutli/KL/

python logitlens.py llama_3_1_8b_instruct --language english --data-path /home/ubuntu/llm_effective_depth/data/english_analysis_cases.json --output-dir output/llama_3_1_8b_instruct/english/mutli/KL/
```