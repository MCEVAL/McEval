export MASTER_ADDR=localhost
export MASTER_PORT=2131
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

COMPLETE_DATA_PATH='<complete data>'



MODEL_DIR='<model dir>'
OUT_DIR='<output dir>'
torchrun --nproc_per_node 8 --master_port 7834 generate.py \
                        --base_model $MODEL_DIR \
                        --data_path $COMPLETE_DATA_PATH \
                        --out_path $OUT_DIR \
                        --maxlen_out 1024 \
                        --batch_size 4



