# eval_correlation

## How to Run

- If test on all datasets
```
python eval_correlation.py --task_name all
```

- If test on a specific dataset
```
python eval_correlation.py --task_name xxx 
# xxx is selected from ["faireval", "mtbench", "llmbar", "instrusum"]
```