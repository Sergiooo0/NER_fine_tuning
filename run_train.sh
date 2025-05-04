#!/bin/bash

MODEL=$1

##### defs ################################################################
MAX_LENGTH=128
BATCH_SIZE=32
NUM_EPOCHS=1
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=2
LOGGING_STEPS=100
LOAD_BEST_MODEL_AT_END=True
SAVE_STRATEGY="no"
TRAIN_FILE="data/ner-es.train.json"
VALIDATION_FILE="data/ner-es.valid.json"
###########################################################################

if [ -z "$MODEL" ]; then
  echo "Sintaxe: $0 <modelo>"
  exit 1
fi

OUTPUT_DIR="models/$(basename ${MODEL})-ner"

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
  --do_train \
  --overwrite_output_dir \
  --load_best_model_at_end ${LOAD_BEST_MODEL_AT_END} \
  --save_strategy ${SAVE_STRATEGY}
