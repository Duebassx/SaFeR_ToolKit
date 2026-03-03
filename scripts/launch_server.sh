export CUDA_VISIBLE_DEVICES=2,3
export NCCL_CUMEM_ENABLE=1

MODEL_PATH="Qwen/Qwen3-VL-32B-Instruct"         

PORT=8200                             

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --tensor-parallel-size 2 \
    --port $PORT \
    --disable-log-requests \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 10240
    # --max-num-seqs 400 \

    
