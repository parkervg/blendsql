#!/bin/bash

for model in "gemma_4b" "gemma_12b"; do
  job="python runner.py blt"
  job+=" ${model} --n-runs 5"
  export JOB=${job}; bash SUBMIT.sh
done