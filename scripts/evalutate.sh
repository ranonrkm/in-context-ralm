#!/bin/bash
#SBATCH --no-requeue
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=inference
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --array=0-2%3


MODEL_NAME=gpt2
OUTPUT_DIR=/data/user_data/rsadhukh/wikitext/results

# i=0
# for stride in 256 128 64
# do
#     strides[$i]=$stride
#     i=$((i+1))
# done

# stride=${strides[$SLURM_ARRAY_TASK_ID]}
# echo $stride
stride=64
RETRIEVAL_FILE=/data/user_data/rsadhukh/wikitext/dense_retrieved_stride${stride}

python3 eval_lm.py \
--model_name $MODEL_NAME \
--dataset_path EleutherAI/wikitext_document_level \
--dataset_name wikitext-103-v1 \
--dataset_split test \
--text_col page \
--output_dir $OUTPUT_DIR \
--stride $stride \
--max_length 1024 \
--normalization_level word \
--retrieved_file $RETRIEVAL_FILE

# python3 eval_lm.py \
#  --model_name $MODEL_NAME \
#  --dataset_path wikitext \
#  --dataset_name wikitext-103-v1 \
#  --dataset_split test \
#  --output_dir $OUTPUT_DIR \
#  --stride 4 \
#  --max_length 1024 \
#  --use_knn --knnlm_index_path /data/user_data/rsadhukh/KNN-LM-experiments/checkpoints/gpt2/index_gpt2_116988150_768.indexed \
#  --knnlm_vals_path /data/user_data/rsadhukh/KNN-LM-experiments/checkpoints/gpt2/dstore_gpt2_116988150_768_vals.npy
