import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
import layers
import preprocess
import numpy as np
from data_util import load_glove

print("tf.__version__:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class BiDAF:

    def __init__(
            self, clen, qlen, emb_size,
            max_char_len=16,
            max_features=5000,
            vocab_size=5000,
            num_highway_layers=2,
            encoder_dropout=0,
            num_decoders=2,
            decoder_dropout=0,
            conv_layers=[],
            embedding_matrix=None
    ):
        """
        双向注意流模型
        :param clen:context 长度
        :param qlen: question 长度
        :param emb_size: 词向量维度
        :param max_features: 词汇表最大数量
        :param num_highway_layers: 高速神经网络的个数 2
        :param encoder_dropout: encoder dropout 概率大小
        :param num_decoders:解码器个数
        :param decoder_dropout: decoder dropout 概率大
        """
        self.clen = clen
        self.qlen = qlen
        self.max_char_len = max_char_len
        self.max_features = max_features
        self.emb_size = emb_size
        self.num_highway_layers = num_highway_layers
        self.encoder_dropout = encoder_dropout
        self.num_decoders = num_decoders
        self.decoder_dropout = decoder_dropout
        self.conv_layers = conv_layers
        self.embedding_matrix = embedding_matrix

    def build_model(self):
        """
        构建模型
        :return:
        """
        # 1 embedding 层
        # TODO：homework：使用glove word embedding（或自己训练的w2v） 和 CNN char embedding
        word_cinn = tf.keras.layers.Input(shape=(self.clen, self.max_char_len), name='word_context_input')
        word_qinn = tf.keras.layers.Input(shape=(self.qlen, self.max_char_len), name='word_question_input')

        char_cinn = tf.keras.layers.Input(shape=(self.clen, self.max_char_clen,), name='char_context_input')
        char_qinn = tf.keras.layers.Input(shape=(self.qlen, self.max_char_qlen,), name='char_question_input')

        word_embedding_layer = tf.keras.layers.Embedding(self.max_features,
                                                         self.emb_size,
                                                         weights=[self.embedding_matrix],
                                                         trainable=False)
        w_cemb = word_embedding_layer(word_cinn)
        w_qemb = word_embedding_layer(word_qinn)

        char_embedding_layer = tf.keras.layers.Embedding(self.max_features,
                                                         self.emb_size,
                                                         embeddings_initializer='uniform')
        c_cemb = char_embedding_layer(char_cinn)
        c_qemb = char_embedding_layer(char_qinn)

        c_cemb_c = self.multi_conv1d(c_cemb)
        c_qemb_c = self.multi_conv1d(c_qemb)

        cemb = tf.keras.layers.Concatenate(axis=2)([c_cemb_c, w_cemb])
        qemb = tf.keras.layers.Concatenate(axis=2)([c_qemb_c, w_qemb])

        for i in range(self.num_highway_layers):
            """
            使用两层高速神经网络
            """
            highway_layer = layers.Highway(name=f'Highway{i}')
            chighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'CHighway{i}')
            qhighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'QHighway{i}')
            cemb = chighway(cemb)
            qemb = qhighway(qemb)

        ## 2. 上下文嵌入层
        # 编码器 双向LSTM
        encoder_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.emb_size,
                recurrent_dropout=self.encoder_dropout,
                return_sequences=True,
                name='RNNEncoder'
            ), name='BiRNNEncoder'
        )

        cencode = encoder_layer(cemb)  # 编码context
        qencode = encoder_layer(qemb)  # 编码question

        # 3.注意流层
        similarity_layer = layers.Similarity(name='SimilarityLayer')
        similarity_matrix = similarity_layer([cencode, qencode])

        c2q_att_layer = layers.C2QAttention(name='C2QAttention')
        q2c_att_layer = layers.Q2CAttention(name='Q2CAttention')

        c2q_att = c2q_att_layer(similarity_matrix, qencode)
        q2c_att = q2c_att_layer(similarity_matrix, cencode)

        # 上下文嵌入向量的生成
        merged_ctx_layer = layers.MergedContext(name='MergedContext')
        merged_ctx = merged_ctx_layer(cencode, c2q_att, q2c_att)

        # 4.模型层
        modeled_ctx = merged_ctx
        for i in range(self.num_decoders):
            decoder_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.emb_size,
                    recurrent_dropout=self.decoder_dropout,
                    return_sequences=True,
                    name=f'RNNDecoder{i}'
                ), name=f'BiRNNDecoder{i}'
            )
            modeled_ctx = decoder_layer(merged_ctx)

        # 5. 输出层
        span_begin_layer = layers.SpanBegin(name='SpanBegin')
        span_begin_prob = span_begin_layer([merged_ctx, modeled_ctx])

        span_end_layer = layers.SpanEnd(name='SpanEnd')
        span_end_prob = span_end_layer([cencode, merged_ctx, modeled_ctx, span_begin_prob])

        output_layer = layers.Combine(name='CombineOutputs')
        out = output_layer([span_begin_prob, span_end_prob])

        inn = [char_cinn, char_qinn, word_cinn, word_qinn]

        self.model = tf.keras.models.Model(inn, out)
        self.model.summary(line_length=128)

        optimizer = tf.keras.optimizers.Adadelta(lr=1e-2)
        self.model.compile(
            optimizer=optimizer,
            loss=negative_avg_log_error,
            metrics=[accuracy]
        )

    def multi_conv1d(self, x_emb):
        def conv1d(emb, kernel_sizes, max_char_len=self.max_char_len):
            pool_out = []
            for kernel_size in kernel_sizes:
                conv = tf.keras.layers.Conv1D(filters=2, kernel_size=[kernel_size], strides=1,
                                              activation='relu')(emb)
                pool = tf.keras.layers.MaxPool1D(pool_size=max_char_len - kernel_size + 1)(conv)
                pool_out.append(pool)

            pool_out = tf.keras.layers.concatenate([p for p in pool_out])
            return pool_out

        words_emb = tf.unstack(x_emb, axis=1)
        vec_list = []
        for word_emb in words_emb:
            conv = conv1d(word_emb, [2, 3, 4])
            vec_list.append(conv)

        char_emb = tf.convert_to_tensor(vec_list)
        char_emb = tf.transpose(char_emb, perm=[1, 0, 2, 3])
        char_emb = tf.squeeze(char_emb, axis=2)

        return char_emb


def negative_avg_log_error(y_true, y_pred):
    """
    损失函数计算
    -1/N{sum(i~N)[(log(p1)+log(p2))]}
    :param y_true:
    :param y_pred:
    :return:
    """

    def sum_of_log_prob(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        begin_prob = y_pred_start[begin_idx]
        end_prob = y_pred_end[end_idx]

        return tf.math.log(begin_prob) + tf.math.log(end_prob)

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    batch_prob_sum = tf.map_fn(sum_of_log_prob, inputs, dtype=tf.float32)

    return -tf.keras.backend.mean(batch_prob_sum, axis=0, keepdims=True)


def accuracy(y_true, y_pred):
    """
    准确率计算
    :param y_true:
    :param y_pred:
    :return:
    """

    def calc_acc(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        start_probability = y_pred_start[begin_idx]
        end_probability = y_pred_end[end_idx]

        return (start_probability + end_probability) / 2.0

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    acc = tf.map_fn(calc_acc, inputs, dtype=tf.float32)

    return tf.math.reduce_mean(acc, axis=0)





if __name__ == '__main__':
    ds = preprocess.Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    train_c_char, train_q_char, train_c_word, train_q_word, train_y = ds.get_dataset('./data/squad/train-v1.1.json')
    test_c_char, test_q_char, test_c_word, test_q_word, test_y = ds.get_dataset('./data/squad/dev-v1.1.json')

    print(train_c_char.shape, train_q_char.shape, train_c_word.shape, train_q_word.shape, train_y.shape)
    print(test_c_char.shape, test_q_char.shape, test_c_word.shape, test_q_word.shape, test_y.shape)

    bidaf = BiDAF(
        clen=ds.max_clen,
        qlen=ds.max_qlen,
        emb_size=100,
        max_char_len=ds.max_char_len,
        max_features=len(ds.charset),
        vocab_size=len(ds.word_list),
        conv_layers = [],
        embedding_matrix=ds.embeddings_matrix()
    )
    bidaf.build_model()
    bidaf.model.fit(
        [train_c_char, train_q_char, train_c_word, train_q_word], train_y,
        batch_size=64,
        epochs=10,
        validation_data=([test_c_char, test_q_char, test_c_word, test_q_word], test_y)
    )
