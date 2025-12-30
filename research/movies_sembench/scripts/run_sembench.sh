#!/bin/bash

for model in "qwen_3b" "qwen_14b"; do
  job="python runner.py blt"
  job+=" ${model}"
  export JOB=${job}; bash SUBMIT.sh
done