export MASTER_ADDR=localhost
export MASTER_PORT=2131
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

MODEL_DIR="<sft model path>"
OUT_DIR="<our dir>"

COMPLETE_DATA_PATH='complete'

torchrun --nproc_per_node 8 --master_port 7834 infer.py \
                        --base_model $MODEL_DIR \
                        --data_path $COMPLETE_DATA_PATH \
                        --out_path $OUT_DIR \
                        --batch_size 4


                        