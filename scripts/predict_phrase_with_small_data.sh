#curl 127.0.0.1:8809/qiye_weixin/wujingjing-predict-phrase-with-small-data-start

export CUDA_VISIBLE_DEVICES=3

data_mode=small_data
output_mode=small_output
checkpoint_id=7191

# bert init model in gs
export BERT_MODEL_DIR=/home/users/wujingjing/tmp/pycharm_project_169/models/bert/chinese_L-12_H-768_A-12

base_dir=/home/users/wujingjing/tmp/pycharm_project_169

echo '生成 base 数据 ...'
mkdir -p $base_dir/$data_mode/phrase_data_train

cat $base_dir/$data_mode/ccks_3_nolabel_data/dev_base.json | python $base_dir/scripts/generate_trigger_train.py train > $base_dir/small_data/phrase_data_train/base.data.test
cp $base_dir/$data_mode/phrase_data_train/base.data.test $base_dir/$data_mode/phrase_data_train/data.test


echo '创建结果文件的路径'
export OUTPUT_DIR=$base_dir/$output_mode/phrase_pipeline_train
export LOCAL_OUTPUT=$base_dir/$output_mode/phrase_pipeline_train_models
mkdir $base_dir/$output_mode/phrase_pipeline_train_models


mkdir -p $OUTPUT_DIR
echo '$output_mode -> ' + $OUTPUT_DIR
mkdir -p $LOCAL_OUTPUT

# conll05 bio data
export BIO_DATA_DIR=$base_dir/$data_mode/phrase_data_train

echo '开始预测数据 ...'
python ./src/trigger_bert.py \
    --task_name="identify_trigger" \
    --use_tpu=False \
    --do_train=False  \
    --do_predict=True  \
    --data_dir=$BIO_DATA_DIR  \
    --vocab_file=$BERT_MODEL_DIR/vocab.txt \
    --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
    --init_checkpoint=$base_dir/output/phrase_pipeline_train_08_26/model.ckpt-$checkpoint_id \
    --do_lower_case=False \
    --max_seq_length=460 \
    --train_batch_size=600 \
    --learning_rate=3e-5 \
    --num_train_epochs=8.0 \
    --output_dir=$OUTPUT_DIR \
    --local_output_dir=$LOCAL_OUTPUT \
    --add_crf=True

echo 'base 数据预测完成'

echo '开始转化数据 ...'

#python $base_dir/scripts/revert_phrase_result.py $base_dir/data/phrase_data_train/data.test \
#    $base_dir/output/phrase_pipeline_train_models/label_test.txt > $base_dir/output/phrase_pipeline_train_models/phrase_predict_sentence.txt

python $base_dir/scripts/revert_phrase_result.py $base_dir/$data_mode/phrase_data_train/data.test \
    $base_dir/$output_mode/phrase_pipeline_train_models/label_test.txt | \
    python $base_dir/scripts/convert_phrase_result_role_input.py > \
    $base_dir/$output_mode/phrase_pipeline_train_models/base_predict.data.test

echo '数据转化成功'

echo '============================================================================='

echo '生成 trans 数据 ...'
mkdir -p $base_dir/$data_mode/phrase_data_train

cat $base_dir/$data_mode/ccks_3_nolabel_data/trans_dev.json | python $base_dir/scripts/generate_trigger_train.py dev > $base_dir/$data_mode/phrase_data_train/trans.data.test
cp $base_dir/$data_mode/phrase_data_train/trans.data.test $base_dir/$data_mode/phrase_data_train/data.test

echo '开始预测数据 ...'
python ./src/trigger_bert.py \
    --task_name="identify_trigger" \
    --use_tpu=False \
    --do_train=False  \
    --do_predict=True  \
    --data_dir=$BIO_DATA_DIR  \
    --vocab_file=$BERT_MODEL_DIR/vocab.txt \
    --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
    --init_checkpoint=$base_dir/output/phrase_pipeline_train_08_26/model.ckpt-$checkpoint_id \
    --do_lower_case=False \
    --max_seq_length=460 \
    --train_batch_size=600   \
    --learning_rate=3e-5   \
    --num_train_epochs=8.0  \
    --output_dir=$OUTPUT_DIR \
    --local_output_dir=$LOCAL_OUTPUT \
    --add_crf=True

echo 'base 数据预测完成'

echo '开始转化数据 ...'

#python $base_dir/scripts/revert_phrase_result.py $base_dir/data/phrase_data_train/data.test \
#    $base_dir/output/phrase_pipeline_train_models/label_test.txt > $base_dir/output/phrase_pipeline_train_models/phrase_predict_sentence.txt

python $base_dir/scripts/revert_phrase_result.py $base_dir/$data_mode/phrase_data_train/data.test \
    $base_dir/$output_mode/phrase_pipeline_train_models/label_test.txt | \
    python $base_dir/scripts/convert_phrase_result_role_input.py > \
    $base_dir/$output_mode/phrase_pipeline_train_models/trans_predict.data.test

echo '数据转化成功'

echo '============================================================================='


echo '开始合并预测结果数据'
mkdir $base_dir/$data_mode/role_data_train
cat $base_dir/$output_mode/phrase_pipeline_train_models/trans_predict.data.test > $base_dir/$data_mode/role_data_train/data.test
cat $base_dir/$output_mode/phrase_pipeline_train_models/base_predict.data.test >> $base_dir/$data_mode/role_data_train/data.test

echo '预测结果数据预测完成'

#curl 127.0.0.1:8809/qiye_weixin/wujingjing-predict-phrase-with-small-data-end