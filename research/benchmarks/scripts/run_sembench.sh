#!/bin/bash

#for model in "qwen_3b" "qwen_14b" "gemma_4b" "gemma_12b"; do
#for model in "qwen_7b" "qwen_14b"; do
for model in "gemma_12b"; do
  job="python runner.py t"
  job+=" ${model} --n-runs 3"
  export JOB=${job}; bash SUBMIT.sh
done