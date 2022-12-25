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

def datasetSplit(slicesGlob='slices_dicom/sagittal/*.dcm', seed=None):
    volumeFiles = dict()
    for file in glob('dataset/IXI-T1/*.gz'):
        index = int(re.match(r"IXI([0-9]+)-.*", os.path.basename(file))[1])
        volumeFiles[index] = file
    volumeFiles = pd.DataFrame.from_dict(volumeFiles, orient='index', columns=['volume'])

    data = pd.read_excel('dataset/IXI.xls', index_col='IXI_ID')
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
        if index in validation.index:
            validation_files.append(file)
        if index in test.index:
            test_files.append(file)

    return train_files, validation_files, test_files

def augmentImage(image, label):
    with tf.device('/CPU:0'):
        #Noise and Dropout
        doDropout = tf.random.uniform(shape=[], minval=0, maxval=1)
        if doDropout < 0.3:
            dropoutRate = tf.random.uniform(shape=[], minval=0, maxval=0.04)
            image = tf.nn.dropout(image, dropoutRate)

        doNoise = tf.random.uniform(shape=[], minval=0, maxval=1)
        if doNoise < 0.3:
            noise = tf.random.uniform(shape=[], minval=0, maxval=0.04)
            image = tf.keras.layers.GaussianNoise(noise)(image, training=True)

        #Blankout and blur
        doCutout = tf.random.uniform(shape=[], minval=0, maxval=1)
        if doCutout < 0.3:
            minx = int(128 * 0.04 / 2)
            miny = int(128 * 0.04 / 2)
            maxx = int(128 * 0.24 / 2)
            maxy = int(128 * 0.24 / 2)

            sizex = tf.math.scalar_mul(2, tf.random.uniform(shape=[], minval=minx, maxval=maxx, dtype=tf.dtypes.int32))
            sizey = tf.math.scalar_mul(2, tf.random.uniform(shape=[], minval=miny, maxval=maxy, dtype=tf.dtypes.int32))
            fill = tf.random.uniform(shape=[], minval=0, maxval=1)

            image = tfa.image.random_cutout(image, mask_size=(sizex, sizey), constant_values=fill)

        doBlur = tf.random.uniform(shape=[], minval=0, maxval=1)
        if doBlur < 0.3:
            image = tfa.image.gaussian_filter2d(image, filter_shape=[3, 3], sigma=0.6, constant_values=0)

    return image, label

def getImage(file_path: tf.Tensor, input_size, output_size):
    with tf.device('/CPU:0'):
        image = tf.io.read_file(file_path)
        image = tfio.image.decode_dicom_image(image, color_dim=True, on_error='strict', dtype=tf.float32)[0]
        image /= (2**32 - 1)

        label = tf.image.resize(image, size=output_size)
        image = tf.image.resize(image, size=input_size)

    return image, label

def createDataset(fileList, batch_size, input_size=(128, 128), output_size=(128, 128), seed=None, shuffle=False, augment=False, buffer_size=None, sample=None):
    with tf.device('/CPU:0'):
        if seed != None:
            tf.random.set_seed(seed)

        if sample != None:
            fileList = random.sample(fileList, sample)
        ds = tf.data.Dataset.from_tensor_slices(fileList)
        ds = ds.map(lambda x: tf.py_function(func=getImage, inp=[x, input_size, output_size], Tout=(tf.float32, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        if buffer_size == None:
            buffer_size = len(fileList)
        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if augment:
            ds = ds.map(lambda image, label: tf.py_function(func=augmentImage, inp=[image, label], Tout=(tf.float32, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
