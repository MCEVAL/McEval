export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"


MODEL_PATH='CodeQwen1.5-7B'

DATA_PATH="<your data path>"

SAVE_PATH='./ckpt/sft-codewen-1.5-7b'
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --model_max_length 5000 \
    --save_steps 10000 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    --gradient_checkpointing True \
    --tf32 True


