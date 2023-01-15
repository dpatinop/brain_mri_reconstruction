from glob import glob
import numpy as np
import os
import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio

def datasetSplit(slicesGlob='slices_dicom/sagittal/*.dcm', seed=None, drop_duplicates=False):
    volumeFiles = dict()
    for file in glob('dataset/IXI-T1/*.gz'):
        index = int(re.match(r"IXI([0-9]+)-.*", os.path.basename(file))[1])
        volumeFiles[index] = file
    volumeFiles = pd.DataFrame.from_dict(volumeFiles, orient='index', columns=['volume'])

    data = pd.read_excel('dataset/IXI.xls', index_col='IXI_ID')
    if drop_duplicates:
        data.drop_duplicates(inplace=True)
    data['volume'] = volumeFiles['volume']
    data = data[data['volume'].notnull()][["SEX_ID (1=m, 2=f)", "AGE", 'volume']]

    bins = pd.IntervalIndex.from_tuples([(0, 25), (25, 65), (65,  np.Inf)], closed="left")
    data['AGE'] = pd.cut(data['AGE'], bins)
    data['stratify'] = str(data['AGE']) + '_' + str(data["SEX_ID (1=m, 2=f)"])

    train, validation, a, b = train_test_split(data, data, train_size=0.8, stratify=data[["stratify"]], random_state=seed)
    validation, test, a, b = train_test_split(validation, validation, train_size=0.66, stratify=validation[["stratify"]], random_state=seed)

    train_files = []
    validation_files = []
    test_files = []
    for file in glob(slicesGlob):
        index = int(re.match(r"IXI([0-9]+)-.*", os.path.basename(file))[1])
        if index in train.index:
            train_files.append(file)
        elif index in validation.index:
            validation_files.append(file)
        elif index in test.index:
            test_files.append(file)

    return train_files, validation_files, test_files

@tf.function
def augmentImage(image, label, cutoutRate=0.3):
    with tf.device('/CPU:0'):
        rnd = tf.random.uniform(shape=[4], minval=0, maxval=1)
        dropout_noise = tf.random.uniform(shape=[2], minval=0, maxval=0.04)

        # Dropout
        if rnd[0] < 0.3:
            image = tf.nn.dropout(image, dropout_noise[0])

        # Noise
        if rnd[1] < 0.3:
            image = tf.keras.layers.GaussianNoise(dropout_noise[1])(image, training=True)

        # Blankout, enable enforced cuts if rate is higher than 1.0
        while rnd[2] < cutoutRate:
            mask_size = tf.math.scalar_mul(2, tf.random.uniform(shape=[2], minval=5, maxval=20, dtype=tf.dtypes.int32))
            image = tfa.image.random_cutout(image, mask_size=mask_size, constant_values=0)
            cutoutRate -= 1

        # Blur
        if rnd[3] < 0.3:
            image = tfa.image.gaussian_filter2d(image, filter_shape=[3, 3], sigma=0.6, constant_values=0)

    return image, label

@tf.function
def getImage(file_path: tf.Tensor, input_size, output_size):
    with tf.device('/CPU:0'):
        image = tf.io.read_file(file_path)
        image = tfio.image.decode_dicom_image(image, color_dim=True, on_error='strict', dtype=tf.float32)[0]
        image /= (2**32 - 1)

        # First resize output image if expected size is different from original image
        if output_size[0] == 256 and output_size[1] == 256:
            label = tf.identity(image)
        else:
            label = tf.image.resize(image, size=output_size)

        # Then resize input image if expecte size is different from output
        if output_size[0] == input_size[0] and output_size[1] == input_size[1]:
            image = tf.identity(label)
        else:
            image = tf.image.resize(image, size=input_size)

    return image, label

def createDataset(fileList, batch_size, input_size=(128, 128), output_size=(128, 128), seed=None, shuffle=False, augment=False, buffer_size=None, sample=None, cutoutRate=0.3):
    if seed != None:
        tf.random.set_seed(seed)
    if sample != None:
        fileList = random.sample(fileList, sample)
    if buffer_size == None:
        buffer_size = len(fileList)

    with tf.device('/CPU:0'):
        ds = tf.data.Dataset.from_tensor_slices(fileList)
        ds = ds.map(lambda x: tf.py_function(func=getImage, inp=[x, input_size, output_size], Tout=(tf.float32, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache(filename='cache/' + str(hash(str(fileList) + str(input_size) + str(output_size))))

        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if augment:
            ds = ds.map(lambda image, label: tf.py_function(func=augmentImage, inp=[image, label, cutoutRate], Tout=(tf.float32, tf.float32)), num_parallel_calls=1)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
