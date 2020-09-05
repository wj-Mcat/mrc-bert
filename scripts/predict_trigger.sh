time=$(date +"%Y-%m-%d_%H-%M-%S")
curl http://quick.shenzhuo.vip:10565/qiye_weixin/wujingjing-predict-phrase-start-${time}

#time=$(date "+%Y%m%d%H%M%S")
#script logs/predict_phrase_$time.log

export CUDA_VISIBLE_DEVICES=3
version=_bert
checkpoint_id=8988

# bert init model in gs

base_dir=/home/users/wujingjing/tmp/pycharm_project_169
#base_dir=/home/mist/ccks
echo '生成 base 数据 ...'
mkdir -p $base_dir/data/trigger_test_data$version

export BERT_MODEL_DIR=$base_dir/models/bert/chinese_L-12_H-768_A-12

# 将所有的预测数据全部都融合到一起，然后一起进行预测
cat $base_dir/data/ccks_3_nolabel_data/dev_base.json | python $base_dir/scripts/generate_trigger_train.py train > $base_dir/data/trigger_train_data/base.data.test
cat $base_dir/data/ccks_3_nolabel_data/trans_dev.json | python $base_dir/scripts/generate_trigger_train.py dev > $base_dir/data/trigger_train_data/trans.data.test
cat $base_dir/data/trigger_train_data/base.data.test $base_dir/data/trigger_train_data/trans.data.test > $base_dir/data/trigger_train_data/data.test

echo '创建结果文件的路径'
export OUTPUT_DIR=$base_dir/output/trigger_output$version
export LOCAL_OUTPUT=$OUTPUT_DIR

mkdir -p $OUTPUT_DIR
echo 'output_dir -> ' + $OUTPUT_DIR
mkdir -p $LOCAL_OUTPUT

# conll05 bio data
export BIO_DATA_DIR=$base_dir/data/trigger_train_data

echo '开始预测数据 ...'
python ./trigger_bert.py \
    --task_name="identify_trigger" \
    --use_tpu=False \
    --do_train=False  \
    --do_predict=True  \
    --data_dir=$BIO_DATA_DIR  \
    --vocab_file=$BERT_MODEL_DIR/vocab.txt \
    --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
    --init_checkpoint=$OUTPUT_DIR/model.ckpt-$checkpoint_id \
    --do_lower_case=False \
    --max_seq_length=460 \
    --train_batch_size=8 \
    --predict_batch_size=60 \
    --learning_rate=3e-5 \
    --num_train_epochs=8.0 \
    --output_dir=$OUTPUT_DIR \
    --local_output_dir=$LOCAL_OUTPUT \
    --add_crf=True

echo '开始转化数据 ...'

#python $base_dir/scripts/revert_phrase_result.py $base_dir/data/phrase_data_train/data.test \
#    $base_dir/output/phrase_pipeline_train_models/label_test.txt > $base_dir/output/phrase_pipeline_train_models/phrase_predict_sentence.txt

echo '保存predict trigger的结果文件'

python $base_dir/scripts/revert_phrase_result.py $base_dir/data/trigger_train_data/data.test $base_dir/output/trigger_output${version}/label_test.txt > $base_dir/output/trigger_output${version}/reverted_trigger_result.txt
#$base_dir/output/trigger_output${version}/reverted_trigger_result.txt
mkdir -p $base_dir/data/role_test_data/role${version}/
python $base_dir/scripts/convert_phrase_result_role_input.py $base_dir/output/trigger_output${version}/reverted_trigger_result.txt > $base_dir/data/role_test_data/role${version}/data.test

echo '数据转化成功'

time=$(date +"%Y-%m-%d_%H-%M-%S")
curl 127.0.0.1:8809/qiye_weixin/wujingjing-predict-trigger-end-${time}
