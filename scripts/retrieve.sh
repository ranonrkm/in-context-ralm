#!/bin/bash
#SBATCH --no-requeue
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=retrieve
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/retrieve_%j.out
#SBATCH --error=logs/retrieve_%j.err
#SBATCH --array=0-3%4

MODEL_NAME=gpt2

i=0
for stride in 256 128 64
do
    strides[$i]=$stride
    i=$((i+1))
done

stride=${strides[$SLURM_ARRAY_TASK_ID]}
echo $stride
RETRIEVAL_FILE=/data/user_data/rsadhukh/wikitext/retrieved_stride${stride}

# python3 prepare_retrieval_data.py \
# --retrieval_type sparse \
# --tokenizer_name $MODEL_NAME \
# --max_length 1024 \
# --dataset_path wikitext \
# --dataset_name wikitext-103-v1 \
# --dataset_split test \
# --text_col page \
# --index_name wikipedia-dpr \
# --forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
# --stride $stride \
# --output_file $RETRIEVAL_FILE \
# --num_tokens_for_query 32 \
# --num_docs 16 

# python3 prepare_retrieval_data.py \
# --retrieval_type dense \
# --tokenizer_name $MODEL_NAME \
# --max_length 1024 \
# --dataset_path wikitext \
# --dataset_name wikitext-103-v1 \
# --dataset_split test \
# --index_name wikipedia-dpr \
# --forbidden_titles_path ralm/retrievers/wikitext103_forbidden_titles.txt \
# --stride $stride \
# --output_file $RETRIEVAL_FILE \
# --query_seq_len 128 \
# --num_docs 16

stride=64
ENC_NAME=facebook/dragon-plus-query-encoder
RETRIEVAL_FILE=/data/user_data/rsadhukh/wikitext/dense_retrieved_stride${stride}

python3 prepare_retrieval_data.py \
--retrieval_type dense \
--model_name $ENC_NAME \
--tokenizer_name $MODEL_NAME \
--index_path /data/user_data/rsadhukh/wikipedia/facebook/dragon-plus-context-encoder/index_OPQ64_256_IVF4096_PQ64.indexed \
--datastore_name wikipedia:20220301.en \
--max_length 1024 \
--dataset_path wikitext \
--dataset_name wikitext-103-v1 \
--dataset_split test \
--text_col text \
--query_seq_len 128 \
--stride $stride \
--output_file $RETRIEVAL_FILE \
--num_docs 16