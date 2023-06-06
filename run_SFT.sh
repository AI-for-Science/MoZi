#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
# DeepSpeed Team
UUID=$(uuidgen)
echo "${UUID}"
RUN_FILE=$(readlink -f "$0")
WORK_DIR=$(dirname "$RUN_FILE")
echo "${WORK_DIR}"

# DeepSpeed Team
OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
  OUTPUT=/home//output/${UUID}
fi

mkdir -p $OUTPUT
#bigscience/bloomz-1b7


mkdir -p "${OUTPUT}"/logs
log_file="${OUTPUT}"/logs/train.txt
exec &> >(tee -a "$log_file")

# CUDA_VISIBLE_DEVICES=4
# /nfs/data6/patent_sft/instruction_general_3026150_conversations.json
# /data6/instruction_patent_20k_conversations
deepspeed SFT.py \
   --sft_only_data_path /BELLE/instruction_ip.json \
   --model_name_or_path /mozi-7b-3m-40k \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 2048 \
   --learning_rate 5e-6 \
   --weight_decay 0.0001 \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --zero_stage 3 \
   --gradient_checkpointing \
   --save_steps 2000 \
   --evaluation_steps 500 \
   --output_dir $OUTPUT
#    &> $OUTPUT/training.log
