# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import csv
import json
import os
import tensorflow as tf

from .modeling import BertConfig, BertModel, get_assignment_map_from_checkpoint
from .optimization import create_optimizer
from .tokenization import FullTokenizer, convert_to_unicode, validate_case_matches_checkpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'ner', "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "test_dir", None,
    "The test directory.")

# Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to evaluate training.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 3, "Total batch size for training.") #梯度下降，每批处理32个样本

flags.DEFINE_integer("eval_batch_size", 3, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 3, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 50,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 50,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

# TPU training
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

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """用于序列分类的训练/测试实例"""

    def __init__(self, text_a, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.label = label


class PaddingInputExample(object):
    """
    当需要使用TPU训练时，eval和predict的数据需要是batch_size的整数倍，此类用于处理这类情况
    Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, # 输入部分：token embedding：表示词向量，第一个词是CLS，分隔词有SEP，是单词本身
                 input_mask, # 输入部分：position embedding：为了令transformer感知词与词之间的位置关系
                 segment_ids, # 输入部分：segment embedding：text_a与text_b的句子关系
                 seq_lengths,
                 label_id,  # 输出部分：标签，对应Y
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_lengths = seq_lengths
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r",encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
            return lines


class NerProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")))

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")))

    def get_test_examples(self, text):
        """See base class."""
        #return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")))
        #return self._create_examples(self._read_json(data_dir))
        return self._create_examples(text)

    def get_labels(self):
        clue_labels = ['education', 'experience','publications','softskills', 'professionalskills']
        return ['O'] + [p + '-' + l for p in ['B', 'M', 'E', 'S'] for l in clue_labels]

    def _create_examples(self, text):
        """See base class."""
        examples = []
        text_a = convert_to_unicode(text)
        #text_a = tokenization.convert_to_unicode(line['text'])
        label = ['O'] * len(text_a)
        '''
        for (i, line) in enumerate(lines):
            #guid = "%s" % (i) if 'id' not in line else line['id']
            text_a = tokenization.convert_to_unicode(line['text'])
            label = ['O'] * len(text_a)
            
            if 'label' in line:
                for l, words in line['label'].items():
                    for word, indices in words.items():
                        for index in indices:
                            if index[0] == index[1]-1:
                                label[index[0]] = 'S-' + l
                            else:
                                label[index[0]] = 'B-' + l
                                label[index[1]-1] = 'E-' + l
                                for i in range(index[0] + 1, index[1]):
                                    label[i] = 'M-' + l
        '''
        examples.append(InputExample(text_a=text_a, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            seq_lengths=0,
            label_id=[0] * max_seq_length,
            is_real_example=False)

    label_map = {} # 构建标签和标签id对应关系
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = example.text_a
    # tokens_a = tokenizer.tokenize(example.text_a)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    '''The convention in albert is:
    (a) For sequence pairs:
     tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
     type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    (b) For single sequences:
     tokens:   [CLS] the dog is hairy . [SEP]
     type_ids: 0     0   0   0  0     0 0
    
    Where "type_ids" are used to indicate whether this is the first
    sequence or the second sequence. The embedding vectors for `type=0` and
    `type=1` were learned during pre-training and are added to the wordpiece
    embedding vector (and position vector). This is not *strictly* necessary
    since the [SEP] token unambiguously separates the sequences, but it makes
    it easier for the model to learn the concept of sequences.
    
    For classification tasks, the first vector (corresponding to [CLS]) is
    used as the "sentence vector". Note that this only makes sense because
    the entire model is fine-tuned.'''

    tokens = []
    segment_ids = []
    label_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(0)
    for i, token in enumerate(tokens_a):
        tokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[example.label[i]])
    tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    seq_lengths = len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        seq_lengths=seq_lengths,
        label_id=label_ids,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer,output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file) # 数据存到tfrecord file里，方便训练时读取

    for (ex_index, example) in enumerate(examples): # 每隔50个打印

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features['seq_lengths'] = create_int_feature([feature.seq_lengths])
        features["label_ids"] = create_int_feature(feature.label_id)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length,is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "seq_lengths": tf.FixedLenFeature([], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn
    


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(albert_config, is_training, input_ids, input_mask, segment_ids,
                 seq_lengths, labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = BertModel(
        config=albert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layers = model.get_all_encoder_layers()
    output_layer = tf.concat([output_layers[albert_config.num_hidden_layers//2-1], output_layers[-1]], axis=-1)

    seq_length = output_layer.shape[1].value
    hidden_size = output_layer.shape[2].value
    output_layer = tf.reshape(output_layer, shape=(-1, hidden_size))

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, shape=(-1, seq_length, num_labels))

        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, labels, seq_lengths)
        loss = tf.reduce_mean(-log_likelihood)

        return (loss, logits)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        #tf.logging.info("*** Features ***")
        #for name in sorted(features.keys()):
        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        seq_lengths = features["seq_lengths"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, seq_lengths, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        #tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op, _ = create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            logging_hook = tf.train.LoggingTensorHook({"total_loss:":total_loss},every_n_iter=10)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [total_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": logits},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def get_result(sentence, label):
    result_words = []
    result_pos = []
    temp_word = []
    temp_pos = ''
    for i in range(min(len(sentence), len(label))):
        if label[i].startswith('O'):
            if len(temp_word) > 0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = []
            temp_pos = ''
        elif label[i].startswith('S-'):
            if len(temp_word)>0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            result_words.append([i, i])
            result_pos.append(label[i].split('-')[1])
            temp_word = []
            temp_pos = ''
        elif label[i].startswith('B-'):
            if len(temp_word)>0:
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = [i]
            temp_pos = label[i].split('-')[1]
        elif label[i].startswith('M-'):
            if len(temp_word)>0:
                temp_word.append(i)
                if temp_pos=='':
                    temp_pos = label[i].split('-')[1]
        else:
            if len(temp_word)>0:
                temp_word.append(i)
                if temp_pos=='':
                    temp_pos = label[i].split('-')[1]
                result_words.append([min(temp_word), max(temp_word)])
                result_pos.append(temp_pos)
            temp_word = []
            temp_pos = ''
    return result_words, result_pos


def get_ner_predict(text,
                    output_dir='ner/output',
                    bert_config_file='ner/bert_config.json',
                    vocab_file='ner/vocab.txt',
                    init_checkpoint='ner/model.ckpt-1500',
                    max_seq_length=512,
                    learning_rate=2e-5,
                    train_batch_size=8,):
    tf.logging.set_verbosity(tf.logging.ERROR)

    # init_checkpoint
    validate_case_matches_checkpoint(True, init_checkpoint)
    # bert_config
    bert_config = BertConfig.from_json_file(bert_config_file)
    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(output_dir)

    processor = NerProcessor()

    label_list = processor.get_labels()
    # vocab-tokenizer
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    
    # 使用TPU
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))


    # 准备训练数据
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    flag = False
    #if FLAGS.do_train:
    if flag:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)


    # 预测
    predict_examples = processor.get_test_examples(text)
    num_actual_predict_examples = len(predict_examples)

    predict_file = os.path.join(output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,max_seq_length, tokenizer,predict_file)
    print("***** Running prediction*****") 

    predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=False)

    result = estimator.predict(input_fn=predict_input_fn)
    for prediction in result:
      probabilities = prediction["probabilities"][1:-1, :]
      example = predict_examples[0]
      text = example.text_a
      label = [label_list[np.argmax(p)] for p in probabilities]
      result_words, result_pos = get_result(example.text_a, label)
      res = [text[pos[0]:pos[1]+1] for pos in result_words if pos[1]-pos[0]>0 and pos[1]<len(text)]
    return res


if __name__=="__main__":
    import time
    a = time.time()
    text = "2020-10~2020-12组员卡尔顿“植系女子力”策划案1.作为小组的一部分完成了卡尔顿品牌“植系”产品的校园营销策划案。 2.主要负责策划案中的消费者分析、营销概念以及创意执行阶段的各项活动策划。"
    res = get_ner_predict(text,max_seq_length=512,learning_rate=2e-5,train_batch_size=8,
              output_dir='output',
              bert_config_file='bert_config.json',
              vocab_file='vocab.txt',
              init_checkpoint='model.ckpt-1500')
    b = time.time()
    print(b-a)
    print(res)

