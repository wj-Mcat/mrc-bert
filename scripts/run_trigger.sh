curl 127.0.0.1:8809/qiye_weixin/wujingjing-run-trigger-bert-start

export CUDA_VISIBLE_DEVICES=2
version=_bert
base_dir=/home/users/wujingjing/tmp/pycharm_project_169

export OUTPUT_DIR=$base_dir/output/trigger_output$version
export LOCAL_OUTPUT=$OUTPUT_DIR

mkdir $OUTPUT_DIR

# bert init model in gs
export BERT_MODEL_DIR=$base_dir/models/bert/chinese_L-12_H-768_A-12
# conll05 bio data
export BIO_DATA_DIR=$base_dir/data/trigger_train_data


mkdir -p $LOCAL_OUTPUT

python trigger_bert.py \
    --task_name="identify_trigger" \
    --use_tpu=False \
    --do_train=True  \
    --do_predict=False  \
    --data_dir=$BIO_DATA_DIR  \
    --vocab_file=$BERT_MODEL_DIR/vocab.txt \
    --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
    --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
    --do_lower_case=False \
    --max_seq_length=460 \
    --train_batch_size=20  \
    --learning_rate=1e-5 \
    --num_train_epochs=5.0  \
    --output_dir=$OUTPUT_DIR \
    --local_output_dir=$LOCAL_OUTPUT \
    --add_crf=True

# MistGpu -> 6

curl 127.0.0.1:8809/qiye_weixin/wujingjing-run-trigger-bert-end