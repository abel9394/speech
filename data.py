# 先将数据分为训练集和验证集
# 再将训练集中全部音频整理为1s
# 最后给出训练参数

# Data Loading
import os
import re
from glob import glob
# Data Generator
import numpy as np
from scipy.io import wavfile
# Parameters
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATADIR = './data/' # unzipped train and test data
OUTDIR = './result/' # just a random name

# 需要识别的单词
POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

# 通过正则表达式找对应的测试集及验证集
# label是识别的单词如zero，label_id为单词对应的序号
# uid为16进制文件名ffd2ba2f, entry1为路径名zero/ffd2ba2f_nohash_4.wav
# all_files为所有文件的全部路径，如'./data/train/audio\\sheila\\a4ca3afe_nohash_0.wav'
def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
    # print(all_files)

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)

        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:

        # # 注意采用正则表达式，路径需要用\\划分
        # entry1 = entry.split('\\')
        # entry1 = entry1[1]+'/'+entry1[2]
        # print(entry1)
        r = re.match(pattern, entry)

        if r:
            label, uid = r.group(2), r.group(3)
            # print(label, uid, entry1)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label_id, uid, entry)
            # print(label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val

trainset, valset = load_data(DATADIR)
# print(valset)

# 在测试集中：舍弃小于1s的音频，大于1s的音频统一整理为1s，开始的位置随机
# 输出：音频的标签
def data_generator(data, params, mode = 'train'):
    def generator():
        if mode == 'train':
            np.random.shuffle(data)
        # Feel free to add any augmentation
        for (label_id, uid, fname) in data:
            try:
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                L = 16000  # be aware, some files are shorter than 1 sec!
                if len(wav) < L:
                    continue
                # let's generate more silence!
                samples_per_file = 1 if label_id != name2id['silence'] else 20
                for _ in range(samples_per_file):
                    if len(wav) > L:
                        start = np.random.randint(0, len(wav) - L)
                    else:
                        start = 0
                    yield dict(
                        target = np.int32(label_id),
                        wav = wav[start: start + L],
                    )
            except Exception as err:
                print(err, label_id, uid, fname)

    return generator

