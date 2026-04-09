from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, MaxPooling2D, Conv2D, Dropout, Lambda, Dense, Flatten, Activation, Input, Embedding, BatchNormalization
from tensorflow.keras.initializers import glorot_normal, Zeros, TruncatedNormal
from tensorflow.keras.regularizers import l2


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy


from tensorflow.keras.optimizers import Adam
from collections import defaultdict
import numpy as np
import tensorflow as tf
import math


class FeaturesEmbedding(Layer):
    def __init__(self, field_dims, embed_dim, **kwargs):
        super(FeaturesEmbedding, self).__init__(**kwargs)
        self.total_dim = sum(field_dims)
        self.embed_dim = embed_dim
        ## 이부분 dtype=np.int64 이거로 바꿔주기
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.embedding = tf.keras.layers.Embedding(input_dim=self.total_dim, output_dim=self.embed_dim)

    def build(self, input_shape):
        self.embedding.build(input_shape)
        self.embedding.set_weights([tf.keras.initializers.GlorotUniform()(shape=self.embedding.weights[0].shape)])

    def call(self, x):
        x = x + tf.constant(self.offsets)
        return self.embedding(x)

class MultiLayerPerceptron(Layer):
    def __init__(self, input_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, init_std=0.0001, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        hidden_units = [input_dim] + list(hidden_units)
        if output_layer:
            hidden_units += [1]

        self.linears = [Dense(units, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=init_std),
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)) for units in hidden_units[1:]]
        self.activation = tf.keras.layers.Activation(activation)
        if self.use_bn:
            self.bn = [BatchNormalization() for _ in hidden_units[1:]]
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.use_bn:
                x = self.bn[i](x, training=training)
            x = self.activation(x)
            x = self.dropout(x, training=training)
        return x

class MultiHeadSelfAttention(Layer):

    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, scaling=False, seed=1024, **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        self.scaling = scaling
        super(MultiHeadSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        embedding_size = int(input_shape[-1])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                     dtype=tf.float32,
                                     initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                       dtype=tf.float32,
                                       initializer=TruncatedNormal(seed=self.seed + 2))
        if self.use_res:
            self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num],
                                         dtype=tf.float32,
                                         initializer=TruncatedNormal(seed=self.seed))

        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))

        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        inner_product = tf.matmul(querys, keys, transpose_b=True)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores =  tf.nn.softmax(inner_product)

        result = tf.matmul(self.normalized_att_scores, values)
        result = tf.concat(tf.split(result, self.head_num, ), axis=-1)
        result = tf.squeeze(result, axis=0) 

        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))
        result = tf.nn.relu(result)

        return result

    def compute_output_shape(self, input_shape):

        return (None, input_shape[1], self.att_embedding_size * self.head_num)

    def get_config(self, ):
        config = {'att_embedding_size': self.att_embedding_size, 'head_num': self.head_num
                  , 'use_res': self.use_res, 'seed': self.seed}
        base_config = super(MultiHeadSelfAttention, self).get_config()
        base_config.update(config)
        return base_config


class AutoIntMLP(Layer): 
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, dnn_hidden_units=(32, 32), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0.4, init_std=0.0001):
        super(AutoIntMLP, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embedding_size)
        self.num_fields = len(field_dims)
        self.embedding_size = embedding_size

        self.final_layer = Dense(1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=init_std))
        
        self.dnn = tf.keras.Sequential()
        for units in dnn_hidden_units:
            self.dnn.add(Dense(units, activation=dnn_activation,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn),
                               kernel_initializer=tf.random_normal_initializer(stddev=init_std)))
            if dnn_use_bn:
                self.dnn.add(BatchNormalization())
            self.dnn.add(Activation(dnn_activation))
            if dnn_dropout > 0:
                self.dnn.add(Dropout(dnn_dropout))
        self.dnn.add(Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=init_std)))

        self.int_layers = [MultiHeadSelfAttention(att_embedding_size=embedding_size, head_num=att_head_num, use_res=att_res) for _ in range(att_layer_num)]

    def call(self, inputs):
        embed_x = self.embedding(inputs)
        dnn_embed = tf.reshape(embed_x, shape=(-1, self.embedding_size * self.num_fields))

        att_input = embed_x
        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = Flatten()(att_input)
        att_output = self.final_layer(att_output)
        
        dnn_output = self.dnn(dnn_embed)
        y_pred = tf.keras.activations.sigmoid(att_output + dnn_output)
        
        return y_pred


class AutoIntMLPModel(Model):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2,
                 att_res=True, dnn_hidden_units=(32, 32), dnn_activation='relu',
                 l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False,
                 dnn_dropout=0.4, init_std=0.0001):
        super(AutoIntMLPModel, self).__init__()
        self.autoInt_layer = AutoIntMLP(
            field_dims=field_dims,
            embedding_size=embedding_size,
            att_layer_num=att_layer_num,
            att_head_num=att_head_num,
            att_res=att_res,
            dnn_hidden_units=dnn_hidden_units,
            dnn_activation=dnn_activation,
            l2_reg_dnn=l2_reg_dnn,
            l2_reg_embedding=l2_reg_embedding,
            dnn_use_bn=dnn_use_bn,
            dnn_dropout=dnn_dropout,
            init_std=init_std
        )

    def call(self, inputs, training=False):
        return self.autoInt_layer(inputs, training=training)
    
    
def predict_model(model, pred_df):
    batch_size = 2048
    top = 10
    results = [] # 결과를 담을 단일 리스트
    
    total_rows = len(pred_df)
    for i in range(0, total_rows, batch_size):
        # 1. 모델 피처 개수가 14개라면 정교하게 슬라이싱 (label 컬럼 제외 등)
        # 만약 pred_df에 피처만 있다면 [i:i + batch_size, :] 가 맞습니다.
        features = pred_df.iloc[i:i + batch_size, :].values 
        
        y_pred = model.predict(features, verbose=False)
        
        for feature, p in zip(features, y_pred):
            # feature[0]은 user_id, feature[1]은 movie_id라고 가정
            i_id = int(feature[1])
            score = float(p.item() if hasattr(p, 'item') else p[0])
            
            # (아이템ID, 점수) 튜플을 리스트에 추가
            results.append((i_id, score))
    
    # 2. 모든 영화에 대한 예측 점수 중 상위 top개를 점수(s[1]) 기준으로 정렬
    return sorted(results, key=lambda s: s[1], reverse=True)[:top]