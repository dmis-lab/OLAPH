#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python pdata_collection.py --model_name_or_path Minbyul/mistral-7b-wo-kqa_golden-iter-sft-step1 --eval_data live_qa
CUDA_VISIBLE_DEVICES=6 python pdata_collection.py --model_name_or_path Minbyul/mistral-7b-wo-kqa_golden-iter-sft-step1 --eval_data medication_qa
CUDA_VISIBLE_DEVICES=6 python pdata_collection.py --model_name_or_path Minbyul/mistral-7b-wo-kqa_golden-iter-sft-step1 --eval_data kqa_golden
CUDA_VISIBLE_DEVICES=6 python pdata_collection.py --model_name_or_path Minbyul/mistral-7b-wo-kqa_golden-iter-sft-step1 --eval_data kqa_silver_wogold
CUDA_VISIBLE_DEVICES=6 python pdata_collection.py --model_name_or_path Minbyul/mistral-7b-wo-kqa_golden-iter-sft-step1 --eval_data healthsearch_qa --data_size to1k
CUDA_VISIBLE_DEVICES=6 python pdata_collection.py --model_name_or_path Minbyul/mistral-7b-wo-kqa_golden-iter-sft-step1 --eval_data healthsearch_qa --data_size 1kto2k
CUDA_VISIBLE_DEVICES=6 python pdata_collection.py --model_name_or_path Minbyul/mistral-7b-wo-kqa_golden-iter-sft-step1 --eval_data healthsearch_qa --data_size 2ktoall