export CUDA_VISIBLE_DEVICES=2

version=_chinese_wwm_ext_L-12-H-768_A-12

export OUTPUT_DIR=/home/users/wujingjing/tmp/pycharm_project_169/output/role_pipeline_train$version
export LOCAL_OUTPUT=/home/users/wujingjing/tmp/pycharm_project_169/output/role_pipeline_train_models$version

mkdir $OUTPUT_DIR
mkdir $LOCAL_OUTPUT

# bert init model in gs
export BERT_MODEL_DIR=/home/users/wujingjing/tmp/pycharm_project_169/models/chinese_wwm_ext_L-12-H-768_A-12
# conll05 bio data
export BIO_DATA_DIR=/home/users/wujingjing/tmp/pycharm_project_169/data/role_data_train


mkdir -p $LOCAL_OUTPUT

python role_bert.py \
    --task_name="identify_role" \
    --use_tpu=False \
    --do_train=True  \
    --do_predict=False  \
    --data_dir=$BIO_DATA_DIR  \
    --vocab_file=$BERT_MODEL_DIR/vocab.txt \
    --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
    --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
    --do_lower_case=False \
    --max_seq_length=460 \
    --train_batch_size=8  \
    --learning_rate=3e-5   \
    --num_train_epochs=5.0  \
    --output_dir=$OUTPUT_DIR \
    --local_output_dir=$LOCAL_OUTPUT \
    --add_crf=True

curl 127.0.0.1:8809/qiye_weixin/wujingjing-train-role-end