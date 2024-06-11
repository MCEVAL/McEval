
MODEL_DIR='<model dir>'

COMPLETE_DATA_PATH='./completion/merge'
python inference_vllm.py \
    --data_path $COMPLETE_DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'completion'  \
    --outdir 'completion_result'


COMPLETE_DATA_PATH='./completion/light'
python inference_vllm.py \
    --data_path $COMPLETE_DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'completion_light'  \
    --outdir 'completion_result'