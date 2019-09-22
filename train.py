# Main Function
# Let's run training!

import tensorflow as tf
# it's a magic function :)
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

from data import *
from convnet import create_model
import os
from glob import glob
import numpy as np
from scipy.io import wavfile


# TensorFlow配置日志记录
tf.logging.set_verbosity(tf.logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 超参数设置
# Define some params. Move model hyperparams (optimizer, extractor, num of layers, activation fn, ...) here
params = dict(
    seed = 20,
    batch_size = 1000,
    keep_prob = 0.5,
    learning_rate = 1e-3,
    clip_gradients = 15.0,
    use_batch_norm = True,
    num_classes = len(POSSIBLE_LABELS),
)

hparams = tf.contrib.training.HParams(**params)
os.makedirs(os.path.join(OUTDIR, 'eval'), exist_ok = True)
model_dir = OUTDIR

run_config = tf.contrib.learn.RunConfig(model_dir = model_dir)


# 数据产生：超参数传递
# 训练集超参数传递
train_input_fn = generator_input_fn(
    x = data_generator(trainset, hparams, 'train'),
    target_key = 'target',  # you could leave target_key in features, so labels in model_handler will be empty
    batch_size = hparams.batch_size, shuffle = True, num_epochs = None,
    queue_capacity = 3 * hparams.batch_size + 10, num_threads = 1
)

# 验证集超参数传递
val_input_fn = generator_input_fn(
    x = data_generator(valset, hparams, 'val'),
    target_key = 'target',
    batch_size = hparams.batch_size, shuffle = True, num_epochs = None)

# 封装模型，超参数声明，训练集及测试集轮数
# 训练集轮数设置：总共50k+数据，batch_size = 1000，完整过一轮全部数据需要50次steps，一般需要过1000轮全部数据，则steps设置为50k
def _create_my_experiment(run_config, hparams):
    exp = tf.contrib.learn.Experiment(
        estimator = create_model(config = run_config, hparams = hparams),
        train_input_fn = train_input_fn,
        eval_input_fn = val_input_fn,
        train_steps = 50000,  # just randomly selected params
        eval_steps = 100,  # read source code for steps-epochs ariphmetics
        train_steps_per_iteration = 1,
    )
    return exp

# 正式开始训练以及验证
tf.contrib.learn.learn_runner.run(
    experiment_fn = _create_my_experiment,
    run_config = run_config,
    schedule = "continuous_train_and_eval",
    hparams = hparams)


# 开始预测
# predict testset
paths = glob(os.path.join(DATADIR, 'test/audio/*wav'))

# 测试集音频数据产生
def test_data_generator(data):
    def generator():
        for path in data:
            _, wav = wavfile.read(path)
            wav = wav.astype(np.float32) / np.iinfo(np.int16).max
            fname = os.path.basename(path)
            yield dict(
                sample = np.string_(fname),
                wav = wav,
            )
    return generator

# 测试集超参数传递
test_input_fn = generator_input_fn(
    x = test_data_generator(paths),
    batch_size = hparams.batch_size,
    shuffle = False,
    num_epochs = 1,
    queue_capacity = 10 * hparams.batch_size,
    num_threads = 1,
)

# 正式开始预测
model = create_model(config = run_config, hparams = hparams)
it = model.predict(input_fn = test_input_fn)
for i in range(10):
    print(next(it))