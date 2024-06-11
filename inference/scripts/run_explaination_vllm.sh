
MODEL_DIR='<model dir>'

COMPLETE_DATA_PATH='./explain_data'
python inference_vllm.py \
    --data_path $COMPLETE_DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'explain_stage1' \
    --outdir 'explain_stage1'

python gen_stage2_instruction.py

COMPLETE_DATA_PATH='./explain_stage2/explain_stage1'
python inference_vllm.py \
    --data_path $COMPLETE_DATA_PATH \
    --base_model $MODEL_DIR \
    --task 'explain_stage2' \
    --outdir 'explain_result' 
