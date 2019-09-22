"""
Spectrograms (input x) have shape (batch_size, time_frames, freq_bins, 2).
Logits is a tensor with shape (batch_size, num_classes).

We need to write a model handler for three regimes: train, eval, predict
Loss function, train_op, additional metrics and summaries should be defined.
Also, we need to convert sound waveform into spectrograms (we could do it with numpy/scipy/librosa in data generator, but TF has new signal processing API)
"""

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib import signal
# import tensorflow.contrib.slim as slim

# 建立简单版本卷积神经网络结构：BN + CONV + ELU + POOL，注意各层输出矩阵维度
def baseline(x, params, is_training):
    x = layers.batch_norm(x, is_training=is_training)
    for i in range(4):
        x = layers.conv2d(
            x, 16 * (2 ** i), 3, 1,
            activation_fn=tf.nn.elu,
            normalizer_fn=layers.batch_norm if params.use_batch_norm else None,
            normalizer_params={'is_training': is_training}
        )
        x = layers.max_pool2d(x, 2, 2)

    # print(x.get_shape().as_list())

    # # just take two kind of pooling and then mix them, why not :)
    mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
    apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    x = 0.5 * (mpool + apool)
    # print(x.get_shape().as_list())

    # 输出：一般为两个全连接层，此处采用为1*1的卷积层输出（非全连接层） + dropout
    # we can use conv2d 1x1 instead of dense
    x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
    x = tf.nn.dropout(x, keep_prob=params.keep_prob if is_training else 1.0)

    # again conv2d 1x1 instead of dense layer
    logits = layers.conv2d(x, params.num_classes, 1, 1, activation_fn=None)
    return tf.squeeze(logits, [1, 2])



# def vgg_16(inputs,
#            num_classes=1000,
#            is_training=True,
#            dropout_keep_prob=0.5,
#            spatial_squeeze=True,
#            scope='vgg_16',
#            fc_conv_padding='VALID',
#            global_pool=False):
#
#   with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
#     end_points_collection = sc.original_name_scope + '_end_points'
#     # Collect outputs for conv2d, fully_connected and max_pool2d.
#     with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
#                         outputs_collections=end_points_collection):
#       net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#       net = slim.max_pool2d(net, [2, 2], scope='pool1')
#       net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#       net = slim.max_pool2d(net, [2, 2], scope='pool2')
#       net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#       net = slim.max_pool2d(net, [2, 2], scope='pool3')
#       net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#       net = slim.max_pool2d(net, [2, 2], scope='pool4')
#       net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#       net = slim.max_pool2d(net, [2, 2], scope='pool5')
#
#       # Use conv2d instead of fully_connected layers.
#       net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
#       net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                          scope='dropout6')
#       net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
#       # Convert end_points_collection into a end_point dict.
#       end_points = slim.utils.convert_collection_to_dict(end_points_collection)
#       if global_pool:
#         net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
#         end_points['global_pool'] = net
#       if num_classes:
#         net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                            scope='dropout7')
#         net = slim.conv2d(net, num_classes, [1, 1],
#                           activation_fn=None,
#                           normalizer_fn=None,
#                           scope='fc8')
#         if spatial_squeeze:
#           net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
#         end_points[sc.name + '/fc8'] = net
#       return net, end_points
# vgg_16.default_image_size = 224



# 声明变量共享范围，列写不同训练集、验证集、测试集中的变量，声音信息变成频谱信息
# features is a dict with keys: tensors from our datagenerator
# labels also were in features, but excluded in generator_input_fn by target_key
def model_handler(features, labels, mode, params, config):
    # Im really like to use make_template instead of variable_scopes and re-usage
    # 采用函数封装，实现变量共享
    extractor = tf.make_template(
        'extractor', baseline,
        create_scope_now_=True,
    )
    # extractor = tf.make_template(
    #     'extractor', vgg_16,
    #     create_scope_now_=True,
    # )

    # wav is a waveform signal with shape (16000, )
    wav = features['wav']
    # we want to compute spectograms by means of short time fourier transform:
    specgram = signal.stft(
        wav,
        400,  # 16000 [samples per second] * 0.025 [s] -- default stft window frame
        160,  # 16000 * 0.010 -- default stride
    )
    # specgram is a complex tensor, so split it into abs and phase parts:
    phase = tf.angle(specgram) / np.pi
    # log(1 + abs) is a default transformation for energy units
    amp = tf.log1p(tf.abs(specgram))

    x = tf.stack([amp, phase], axis=3)  # shape is [bs, time, freq_bins, 2]
    x = tf.to_float(x)  # we want to have float32, not float64

    # 训练输出变量声明
    logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)
    global_step = tf.train.global_step

    # 训练集中求变量loss
    if mode == tf.estimator.ModeKeys.TRAIN:

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        logging_hook = tf.train.LoggingTensorHook({"loss": loss
                                                   }, every_n_iter=1)

        # some lr tuner, you could use move interesting functions
        def learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate, global_step, decay_steps=10000, decay_rate=0.99)

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params.learning_rate,
            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
            learning_rate_decay_fn=learning_rate_decay_fn,
            clip_gradients=params.clip_gradients,
            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        specs = dict(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )

    # 验证集变量
    if mode == tf.estimator.ModeKeys.EVAL:
        prediction = tf.argmax(logits, axis=-1)
        acc, acc_op = tf.metrics.mean_per_class_accuracy(
            labels, prediction, params.num_classes)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        logging_hook = tf.train.LoggingTensorHook({"loss": loss
                                                   }, every_n_iter=1)
        specs = dict(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                acc=(acc, acc_op),
            )
        )

    # 测试集变量
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'label': tf.argmax(logits, axis=-1),  # for probability just take tf.nn.softmax()
            'sample': features['sample'],  # it's a hack for simplicity
        }
        logging_hook = tf.train.LoggingTensorHook({"prediction": predictions['sample']
                                                   }, every_n_iter=1)
        specs = dict(
            mode=mode,
            predictions=predictions,
        )
    return tf.estimator.EstimatorSpec(**specs, training_hooks=[logging_hook])

# 封装整个模型
def create_model(config=None, hparams=None):
    return tf.estimator.Estimator(
        model_fn=model_handler,
        config=config,
        params=hparams
    )

