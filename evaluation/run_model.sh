USE_XFT=1 # 0: use pytorch to infer, default use xFT to infer

TOKEN_PATH=/data/models/Llama-2-7b-chat-hf/
TRUST_REMOTE_CODE=False # True for chatglm*, baichuan
TASKS="lambada_openai" # boolq,piqa,lambada_openai/standard hellaswag,winogrande,boolq,piqa,arc_challenge,arc_easy,openbookqa
DATA_FILES=${PWD}/lambada_test.jsonl
NUM_FEWSHOT=0
LIMIT=0.05 # if < 1, is a percentage of the total number of the dataset
BATCH_SIZE=16

rm -fr ./lm_cache/*

if [ "${USE_XFT}" -eq 1 ]; then

MODEL_PATH=/data/models/Llama-2-7b-chat-cpu/
MODEL_TYPE=llama # llama gpt(for opt) chatglm chatglm2/3 baichuan
DTYPE=bf16 #fp16, bf16, int8, int4
KVCacheTYPE=fp16 #fp16, int8

FIRST_TOKEN_WEIGHT_LOCATION=$1 NEXT_TOKEN_WEIGHT_LOCATION=$2 numactl -N $1 -m $2 python eval.py \
    --model ${MODEL_TYPE} \
    --model_args pretrained=${TOKEN_PATH},weights=${MODEL_PATH},trust_remote_code=${TRUST_REMOTE_CODE},dtype=${DTYPE},kvtype=${KVCacheTYPE} \
    --tasks ${TASKS} \
    --num_fewshot ${NUM_FEWSHOT} \
    --limit ${LIMIT} \
    --batch_size ${BATCH_SIZE} \
    --data_files ${DATA_FILES} \
    --device cpu

else

MODEL_TYPE=hf
DTYPE=bfloat16 #float bfloat16 auto

export TOKENIZERS_PARALLELISM=false
numactl -N $1 -m $2 python eval.py \
    --model ${MODEL_TYPE} \
    --model_args pretrained=${TOKEN_PATH},trust_remote_code=${TRUST_REMOTE_CODE},dtype=${DTYPE} \
    --tasks ${TASKS} \
    --num_fewshot ${NUM_FEWSHOT} \
    --limit ${LIMIT} \
    --batch_size ${BATCH_SIZE} \
    --data_files ${DATA_FILES} \
    --device cpu

fi
