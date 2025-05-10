#!/bin/bash

MODEL="dccuchile/bert-base-spanish-wwm-cased"

##### defs ################################################################
MAX_LENGTH=128
BATCH_SIZE=32
NUM_EPOCHS=3
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=100
LOAD_BEST_MODEL_AT_END="true"
   #new parameters to save the best model
EVAL_STRATEGY="steps" 
SAVE_STRATEGY="steps"
METRIC_FOR_BEST_MODEL="f1"
GREATER_IS_BETTER="true" 
  #########################################
TRAIN_FILE="data/ner-es-complete.train.jsonl"
VALIDATION_FILE="data/ner-es.valid.jsonl"
###########################################################################

if [ -z "$MODEL" ]; then
  echo "Sintaxe: $0 <modelo>"
  exit 1
fi

OUTPUT_DIR="models/$(basename ${MODEL})-ner"

echo "Eval strategy: $EVAL_STRATEGY"
echo "Save strategy: $SAVE_STRATEGY"
echo "Load best model: $LOAD_BEST_MODEL_AT_END"
echo "Metric for best model: $METRIC_FOR_BEST_MODEL"

# https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
echo "Starting training: ${MODEL}"
time python3 run_ner.py \
  --model_name_or_path ${MODEL} \
  --train_file ${TRAIN_FILE} \
  --validation_file ${VALIDATION_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --max_seq_length ${MAX_LENGTH} \
  --num_train_epochs ${NUM_EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit ${SAVE_TOTAL_LIMIT} \
  --logging_steps ${LOGGING_STEPS} \
  --eval_strategy ${EVAL_STRATEGY} \
  --save_strategy ${SAVE_STRATEGY} \
  --load_best_model_at_end ${LOAD_BEST_MODEL_AT_END} \
  --metric_for_best_model ${METRIC_FOR_BEST_MODEL} \
  --greater_is_better ${GREATER_IS_BETTER} \
  --do_train \
  --do_eval \
  --overwrite_output_dir
  
