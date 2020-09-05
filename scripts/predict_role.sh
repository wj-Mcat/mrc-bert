time=$(date "+%Y%m%d%H%M%S")
script logs/predict_role_$time.log

curl 127.0.0.1:8809/qiye_weixin/wujingjing-predict-role-start

export CUDA_VISIBLE_DEVICES=3

base_dir=/home/users/wujingjing/tmp/pycharm_project_169
version=_bert_08_28

export OUTPUT_DIR=$base_dir/output/role_pipeline_train$version
export LOCAL_OUTPUT=$base_dir/output/role_pipeline_train_models$version

# bert init model in gs
export BERT_MODEL_DIR=$base_dir/models/bert/chinese_L-12_H-768_A-12
# conll05 bio data
export BIO_DATA_DIR=$base_dir/data/role_data_train$version

mkdir -p $LOCAL_OUTPUT

echo '开始预测role结果'

python ./role_bert.py \
    --task_name="identify_role" \
    --use_tpu=False \
    --do_train=False  \
    --do_predict=True  \
    --data_dir=$BIO_DATA_DIR  \
    --vocab_file=$BERT_MODEL_DIR/vocab.txt \
    --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
    --init_checkpoint=$OUTPUT_DIR/model.ckpt-29000 \
    --do_lower_case=False \
    --max_seq_length=460 \
    --train_batch_size=60 \
    --predict_batch_size=60 \
    --learning_rate=3e-5   \
    --num_train_epochs=8.0  \
    --output_dir=$OUTPUT_DIR \
    --local_output_dir=$LOCAL_OUTPUT \
    --add_crf=True



echo '合并原始数据...'
cat data/ccks_3_nolabel_data/dev_base.json > data/ccks_3_nolabel_data/dev.json
cat data/ccks_3_nolabel_data/trans_dev.json >> data/ccks_3_nolabel_data/dev.json

echo '转化预测结果文件...'
python $base_dir/scripts/merge_role_result.py

curl 127.0.0.1:8809/qiye_weixin/wujingjing-predict-role-end
