#!/bin/bash

for model in "gemma_4b" "gemma_12b" "qwen_4b" "qwen_14b"; do
  job="python runner.py blt"
  job+=" ${model}"
  export JOB=${job}; bash SUBMIT.sh
done