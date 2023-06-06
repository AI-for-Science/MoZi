#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import json
import os

import deepspeed
import jsonlines
import math
import torch
import wandb
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaTokenizer
)

from bnnt.data.data_utils import get_shuffle_idx
from bnnt.data.raw_datasets import BNNTDataset
from bnnt.ds_utils import get_train_ds_config
from bnnt.model.model_utils import create_hf_model
from bnnt.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters
from bnnt.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                             '1) a single data path, 2) multiple datasets in the'
                             'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='6,2,2',
                        help='Comma-separated list of proportions for training'
                             'phase 1, 2, and 3 data. For example the split `2,4,4`'
                             'will use 60% of data for phase 1, 20% for phase 2'
                             'and 20% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.1,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--distributed_port",
                        type=int,
                        default=-1,
                        help="distributed_port for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--save_steps',
                        type=int,
                        default=10,
                        help='Num of steps to save checkpoint.')
    parser.add_argument('--evaluation_steps',
                        type=int,
                        default=10,
                        help='Num of steps to evaluate.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."

    return args


def main():
    args = parse_args()

    # world_size = torch.distributed.get_world_size()
    PROJECT_NAME = f"bloomz-7b1-gpu-8-bs-{args.per_device_train_batch_size}-grad-accum-{args.gradient_accumulation_steps}-lr-{args.learning_rate}-maxlen-{args.max_seq_len}-ZeRO-{args.zero_stage}"

    wandb.init(
        # set the wandb project where this run will be logged
        entity="bnnt",
        project=PROJECT_NAME,

        # track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "dataset": "patent-full",
            "Epochs": args.num_train_epochs,
        }
    )

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed(distributed_port=args.distributed_port)

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    torch.distributed.barrier()

    print("model_name_or_path : ", args.model_name_or_path)
    if "llama" in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, cache_dir='/data6/.cache/huggingface/hub/')
        assert tokenizer.eos_token_id == 2
        assert tokenizer.bos_token_id == 1
        args.lora_module_name = [
            "q_proj",
            "k_proj",
            "v_proj",
            "down_proj",
            "gate_proj",
            "up_proj"
        ]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    print("Making tokenizer padding side to left")

    model = create_hf_model(AutoModelForCausalLM, args.model_name_or_path,
                            tokenizer, ds_config)

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        # model = convert_LLaMA_to_lora(model, args.lora_module_name)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # Prepare the data
    train_phase = 1
    # print("sft_only_data_path : ", args.sft_only_data_path)
    # train_dataset, eval_dataset = create_prompt_dataset(
    #     args.local_rank,
    #     args.data_path,
    #     args.data_split,
    #     args.data_output_path,
    #     train_phase,
    #     args.seed,
    #     tokenizer,
    #     args.max_seq_len,
    #     sft_only_data_path=args.sft_only_data_path)

    data_path = f"/data6/xjy2023/xl/new_shard_{args.local_rank}.jsonl"
    print("data_path : ", data_path)
    if data_path.endswith("json"):
        with open(data_path) as f:
            raw_dataset = json.load(f)
            shuffle_idx = get_shuffle_idx(args.seed, len(raw_dataset[:1000]))

            eval_idx = list(shuffle_idx)[:500]
            print(eval_idx)

        print("Loading dataset")
        pbar = tqdm(total=len(raw_dataset), position=args.local_rank)
        train_dataset, eval_dataset = [], []
        # for i, tmp_data in tqdm(enumerate(raw_dataset), total=len(raw_dataset), position=args.local_rank):
        tmp_idx = 0
        while raw_dataset:
            tmp_data = raw_dataset.pop()
            # tokenize the text
            chosen_sentence = f"标题：{tmp_data['title']}。摘要：{tmp_data['summary']}专利公开号：{tmp_data['publicNo']}。权利要求：{tmp_data['powerRequirements']}说明书：{tmp_data['instructions']}"
            chosen_token = tokenizer(chosen_sentence,
                                     max_length=args.max_seq_len,
                                     padding="max_length",
                                     truncation=True)

            chosen_token["input_ids"][-1] = tokenizer.eos_token_id

            chosen_token["labels"] = torch.LongTensor(
                [-100] + [-100 if tokenizer.pad_token_id == idx else idx for idx in chosen_token["input_ids"]][1:])

            chosen_token["input_ids"] = torch.LongTensor(chosen_token["input_ids"]).squeeze(0)
            chosen_token["attention_mask"] = torch.LongTensor(chosen_token["attention_mask"]).squeeze(0)
            chosen_token["labels"] = torch.LongTensor(chosen_token["labels"]).squeeze(0)

            if tmp_idx in eval_idx:
                eval_dataset.append(chosen_token)
            else:
                train_dataset.append(chosen_token)
            tmp_idx += 1
            pbar.update(1)
    elif data_path.endswith("jsonl"):
        # raw_dataset = []
        # with open(data_path,'r',encoding='utf-8') as f:
        #     for _ in tqdm(range(3500000)):
        #         line = f.readline()
        #         line_dict = ast.literal_eval(line)
        #         raw_dataset.append(line_dict)
        total_size = 3500000
        # shuffle_idx = get_shuffle_idx(args.seed, total_size)
        # print("raw dataset: ", total_size)
        # eval_idx = list(shuffle_idx)[:500]
        # print(eval_idx)
        raw_data = []
        with jsonlines.open(data_path) as f:
            for i, tmp_data in tqdm(enumerate(f),
                                    desc=data_path,
                                    position=args.local_rank,
                                    total=total_size):
                if i < total_size:
                    raw_data.append(tmp_data)
                else:
                    break

        train_dataset = BNNTDataset(raw_data[500:], tokenizer, args)
        eval_dataset = BNNTDataset(raw_data[:500], tokenizer, args)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), unit="batch"):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        model.train()
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"ppl: {perplexity}", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch"):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            print("Epoch: {}, step: {}, loss: {}".format(epoch, step, loss.item()))
            wandb.log({"loss": loss.item()}, step=step)

            if (step + 1) % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"Epoch-{epoch + 1}-step-{step + 1}")

                save_zero_three_model(model,
                                      args.global_rank,
                                      save_path,
                                      zero_stage=args.zero_stage)

                print_rank_0(
                    f"Saving checkpoint... Steps {step + 1} Epoch {epoch + 1}/{args.num_train_epochs}",
                    args.global_rank)

            if (step + 1) % args.evaluation_steps == 0:
                perplexity = evaluation(model, eval_dataloader)

                wandb.log({"perplexity": perplexity}, step=step)

                print_rank_0(f"ppl: {perplexity}", args.global_rank)
                print_rank_0(
                    f"***** Evaluating perplexity, Steps {step + 1}, Epoch {epoch + 1}/{args.num_train_epochs} *****",
                    args.global_rank)

        # Evaluate perplexity on the validation set.
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
        print_rank_0(
            f"***** Evaluating perplexity, Steps {step + 1}, Epoch {epoch + 1}/{args.num_train_epochs} *****",
            args.global_rank)

        # Save after each epoch.

        model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        # model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
