#! usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle
import random

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

from bert import modeling
from bert import optimization
from bert import tokenization
import logging, datetime


def set_file_logger():
    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create file handler which logs even debug messages
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S-trigger")
    fh = logging.FileHandler(f'./logs/tensorflow-{time_str}.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

set_file_logger()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", None, "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The gs output directory where the model checkpoints will be written."
)

flags.DEFINE_string(
    "local_output_dir", None,
    "The local output directory where the model checkpoints will be written."
)

# BERT
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

# TPU setting
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

# train eval predict
flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

# model parameters
flags.DEFINE_integer(
    "max_seq_length", 460,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_integer("train_batch_size", 10, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 16, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")

flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate')

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

tf.flags.DEFINE_string("exp_type", "ccks_identify_role", "")

flags.DEFINE_bool("add_lstm", False, "Whether to add bi-lstm on the top")

flags.DEFINE_bool("add_crf", False, "Whether to add crf on the top")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, event_type, role, role_label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        # self.label = label
        self.event_type = event_type
        self.role = role
        self.role_label = role_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask,
                 segment_ids, event_type_mask, trigger_mask, role_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.event_type_mask = event_type_mask
        self.trigger_mask = trigger_mask
        self.role_mask = role_mask
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    # @classmethod
    def _read_data(self, input_file):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            for line in f:
                words = line.strip().split('\t')[1]
                labels = line.strip().split('\t')[0]
                lines.append((labels, words))
            return lines


class IdentifyRoleProcessor(DataProcessor):
    def _read_data(self, input_file):
        with open(input_file) as f:
            lines = []
            for line in f:
                items = line.strip('\n').split('\t')
                sent, event_type, role = items[0], items[1], items[2]
                if len(items) < 4 or items[3] == ' ':
                    role_label = ''
                else:
                    role_label = items[3]

                lines.append((sent.replace(' ', '_'), event_type, role, role_label))

            return lines

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "data.train")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "data.dev")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "data.test")), "test"
        )

    def get_labels(self, exp_type=None):
        if exp_type == "ccks_identify_role":
            return ['O', 'B', 'I']
        else:
            return []

    @staticmethod
    def _create_example(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            event_type = tokenization.convert_to_unicode(line[1])
            role = tokenization.convert_to_unicode(line[2])
            role_label = line[3]
            examples.append(InputExample(guid, text, event_type, role, role_label))
        return examples


def write_tokens(tokens, mode):
    if mode == "test":
        # full output
        path = os.path.join(FLAGS.local_output_dir, "token_" + mode + ".txt")
        wf = open(path, 'a')
        output = " ".join(filter(lambda x: x != "**NULL**", tokens))
        wf.write(output + '\n')
        #for token in tokens:
        #    if token != "**NULL**":
        #        wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, mode):

    text_list = list(example.text)
    label_list = ['O'] * len(text_list)

    trigger_start = example.text.index("<s>") + 3 + 1
    trigger_end = example.text.index("</s>") + 1

    event_type = example.event_type

    role = example.role
    role_labels = example.role_label.replace(' ', '').split(';')

    for role_label in role_labels:
        if not role_label:
            continue
        word, start_pos, end_pos = role_label.split('|')
        start_pos = int(start_pos)
        end_pos = int(end_pos)
        label_list[start_pos] = 'B'
        for i in range(start_pos + 1, end_pos):
            label_list[i] = 'I'
    assert len(text_list) == len(label_list)
        
    # construct bert format token list
    tokens = []
    labels = []
    for i, word in enumerate(text_list):
        label = label_list[i]
        labels.append(label)
        token = tokenizer.tokenize(word)
        if len(token) > 0:
            tokens.extend(token)
        else:
            tokens.append('_')

    assert len(tokens) == len(labels)

    if len(tokens) >= max_seq_length - 20:
        return None

    # fill normal BIO resources
    ntokens = []
    segment_ids = []
    label_ids = []
    # add tokens
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["O"])
    # sent
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # SEP
    ntokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["O"])

    valid_labeling_len = len(ntokens)
    event_type_mask = [0] * valid_labeling_len
    role_mask = [0] * valid_labeling_len

    # add event_type
    event_type_token = tokenizer.tokenize(event_type)
    for token in event_type_token:
        ntokens.append(token)
        segment_ids.append(1)
        label_ids.append(label_map["O"])
        event_type_mask.append(1)
        role_mask.append(0)

    # add ;
    ntokens.append(";")
    segment_ids.append(1)
    label_ids.append(label_map["O"])
    event_type_mask.append(0)
    role_mask.append(0)

    # add role
    role_token = tokenizer.tokenize(role)
    for token in role_token:
        ntokens.append(token)
        segment_ids.append(1)
        label_ids.append(label_map["O"])
        event_type_mask.append(0)
        role_mask.append(1)

    # SEP
    ntokens.append("[SEP]")
    segment_ids.append(1)
    label_ids.append(label_map["O"])
    event_type_mask.append(0)
    role_mask.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)

    trigger_mask = [0] * len(input_ids)
    for i in range(trigger_start, trigger_end):
        trigger_mask[i] = 1

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        event_type_mask.append(0)
        role_mask.append(0)
        trigger_mask.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(event_type_mask) == max_seq_length
    assert len(role_mask) == max_seq_length
    assert len(trigger_mask) == max_seq_length
    # end of fill normal BIO resources

    if ex_index < 50 and random.random() < 0.2:
        print('')
        # tf.logging.info("*** Example ***")
        # tf.logging.info("guid: %s" % example.guid)
        # tf.logging.info("tokens: %s" % " ".join(
        #     [tokenization.printable_text(x) for x in tokens]))
        # tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # tf.logging.info("event_type_mask: %s" % " ".join([str(x) for x in event_type_mask]))
        # tf.logging.info("trigger_mask: %s" % " ".join([str(x) for x in trigger_mask]))
        # tf.logging.info("role_mask: %s" % " ".join([str(x) for x in role_mask]))
        # tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        # tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        event_type_mask=event_type_mask,
        trigger_mask=trigger_mask,
        role_mask=role_mask,
        label_ids=label_ids,
    )

    write_tokens(ntokens, mode)
    return feature


def filed_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, mode=None
):
    # build label_map
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i

    # dump label2id map for later procedure
    with open(os.path.join(FLAGS.local_output_dir, 'label2id.pkl'), 'wb') as w:
        pickle.dump(label_map, w)

    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, mode)
        if feature is None:
            continue

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["event_type_mask"] = create_int_feature(feature.event_type_mask)
        features["trigger_mask"] = create_int_feature(feature.trigger_mask)
        features["role_mask"] = create_int_feature(feature.role_mask)
        features["label_ids"] = create_int_feature(feature.label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "event_type_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "trigger_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "role_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=20000)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, event_type_mask, trigger_mask, role_mask, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    # batch * seq_len * hidden_size
    sent_features = model.get_sequence_output()

    # event type features, mean pooling of each character in the event type word
    event_type_mask = tf.cast(event_type_mask, tf.float32)
    event_type_len = tf.reduce_sum(event_type_mask, axis=-1, keep_dims=True)
    event_type_features = tf.einsum("blh,bl->bh", sent_features, event_type_mask) / event_type_len
    event_type_features = tf.tile(event_type_features[:, None], [1, FLAGS.max_seq_length, 1])

    # role features, mean pooling of each character in the role word
    role_mask = tf.cast(role_mask, tf.float32)
    role_len = tf.reduce_sum(role_mask, axis=-1, keep_dims=True)
    role_features = tf.einsum("blh,bl->bh", sent_features, role_mask) / role_len
    role_features = tf.tile(role_features[:, None], [1, FLAGS.max_seq_length, 1])

    # trigger features, mean pooling of each character in the trigger word
    #trigger_mask = tf.cast(trigger_mask, tf.float32)
    #trigger_len = tf.reduce_sum(trigger_mask, axis=-1, keep_dims=True)
    #trigger_features = tf.einsum("blh,bl->bh", sent_features, trigger_mask) / trigger_len
    #trigger_features = tf.tile(trigger_features[:, None], [1, FLAGS.max_seq_length, 1])

    # final_input = sent_features
    #final_input = tf.concat([sent_features, event_type_features, trigger_features, role_features], axis=-1)
    final_input = tf.concat([sent_features, event_type_features, role_features], axis=-1)

    if FLAGS.add_crf:
        logits = tf.layers.dense(final_input, num_labels,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name="dense_layer")
        trans = tf.get_variable(
            "transitions",
            [num_labels, num_labels],
            initializer=initializers.xavier_initializer()
        )
        sequence_lengths = tf.cast(tf.reduce_sum(input_mask, axis=-1), tf.int32)
        log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=labels,
            transition_params=trans,
            sequence_lengths=sequence_lengths
        )
        loss = tf.reduce_mean(-log_likelihood)

        pred_ids, _ = tf.contrib.crf.crf_decode(potentials=logits, transition_params=trans,
                                                sequence_length=sequence_lengths)
        return loss, pred_ids
    elif FLAGS.add_lstm:
        pass
    else:
        valid_label_num = tf.cast(tf.reduce_sum(input_mask), tf.float32)
        input_mask = tf.cast(input_mask, dtype=tf.float32)

        # dense layer
        mlp = tf.layers.dense(final_input, 768//2, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                              name='mlp', activation=tf.nn.tanh)
        # shape of logits, batch * seq_len * num_labels
        logits = tf.layers.dense(mlp, num_labels, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name="dense_layer")

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        loss *= input_mask

        loss = tf.reduce_sum(loss) / valid_label_num

        probabilities = tf.math.softmax(logits, axis=-1)
        pred_ids = tf.math.argmax(probabilities, axis=-1)
        return loss, pred_ids


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        event_type_mask = features["event_type_mask"]
        trigger_mask = features["trigger_mask"]
        role_mask = features["role_mask"]
        label_ids = features["label_ids"]

        (total_loss, pred_ids) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            event_type_mask, trigger_mask, role_mask,
            num_labels, use_one_hot_embeddings
        )

        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(loss):
                return {
                    "eval_loss": loss,
                }
            eval_metrics = (metric_fn, [total_loss])
            # eval_metrics = (metric_fn, [label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions={"predictions": pred_ids}, scaffold_fn=scaffold_fn
            )
        return output_spec
    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "identify_role": IdentifyRoleProcessor,
    }
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    processor = processors[task_name]()

    label_list = processor.get_labels(FLAGS.exp_type)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list)+1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.local_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_predict:
        token_path = os.path.join(FLAGS.local_output_dir, "token_test.txt")
        with open(os.path.join(FLAGS.local_output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn, yield_single_examples=True)
        output_predict_file = os.path.join(FLAGS.local_output_dir, "label_test.txt")
        with open(output_predict_file, 'w') as writer:
            for prediction in result:
                prediction = prediction["predictions"]
                output_line = " ".join(map(str, [id2label[i] for i in prediction if i != 0])) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()