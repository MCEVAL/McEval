
COMPLETE_DATA_PATH='<generation>'

MODEL_DIR='<model dir>'

python inference_vllm.py \
    --data_path $COMPLETE_DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'generation'  