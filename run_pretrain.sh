#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
UUID=$(uuidgen)
echo "${UUID}"
RUN_FILE=$(readlink -f "$0")
WORK_DIR=$(dirname "$RUN_FILE")
echo "${WORK_DIR}"

# DeepSpeed Team
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
  OUTPUT=/data6/output/${UUID}
fi

mkdir -p $OUTPUT
#bigscience/bloomz-1b7

DISTRIBUTED_PORT=25002

mkdir -p "${OUTPUT}"/logs
log_file="${OUTPUT}"/logs/train.txt
exec &> >(tee -a "$log_file")

PYTHONPATH="${WORK_DIR}"/src deepspeed --master_port 25003 patent_pretrain.py.py \
  --sft_only_data_path belleMath.json \
  --model_name_or_path /data6/.cache/huggingface/hub/models--bigscience--bloomz-7b1-mt/snapshots/13e9b1a39fe86c8024fe15667d063aa8a3e32460/ \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --max_seq_len 2048 \
  --learning_rate 5e-6 \
  --weight_decay 0.0001 \
  --num_train_epochs 1 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 1000 \
  --save_steps 2500 \
  --evaluation_steps 1000 \
  --zero_stage 3 \
  --seed 1234 \
  --gradient_checkpointing \
  --distributed_port $DISTRIBUTED_PORT \
  --output_dir $OUTPUT
#    &> $OUTPUT/training.log

#    --deepspeed_config ds_config.json \
