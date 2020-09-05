export CUDA_VISIBLE_DEVICES=3

base_dir=/home/users/wujingjing/tmp/pycharm_project_169

mkdir $base_dir/small_output/role_pipeline_train
export OUTPUT_DIR=$base_dir/small_output/role_pipeline_train
export LOCAL_OUTPUT=$base_dir/small_output/role_pipeline_train_models
mkdir $base_dir/small_output/role_pipeline_train_models

# bert init model in gs
export BERT_MODEL_DIR=$base_dir/models/bert/chinese_L-12_H-768_A-12
# conll05 bio data
export BIO_DATA_DIR=$base_dir/small_data/role_data_train


mkdir -p $LOCAL_OUTPUT

echo '开始预测role结果'

#
#python ./src/role_bert.py \
#    --task_name="identify_role" \
#    --use_tpu=False \
#    --do_train=False  \
#    --do_predict=True  \
#    --data_dir=$BIO_DATA_DIR  \
#    --vocab_file=$BERT_MODEL_DIR/vocab.txt \
#    --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
#    --init_checkpoint=$base_dir/output/role_pipeline_train_08_27/model.ckpt-25765 \
#    --do_lower_case=False \
#    --max_seq_length=460 \
#    --train_batch_size=60   \
#    --learning_rate=3e-5   \
#    --num_train_epochs=8.0  \
#    --output_dir=$OUTPUT_DIR \
#    --local_output_dir=$LOCAL_OUTPUT \
#    --add_crf=True



echo '合并原始数据...'
cat small_data/ccks_3_nolabel_data/dev_base.json > small_data/ccks_3_nolabel_data/dev.json
cat small_data/ccks_3_nolabel_data/trans_dev.json >> small_data/ccks_3_nolabel_data/dev.json

echo '转化预测结果文件...'
python $base_dir/scripts/merge_role_result.py small_