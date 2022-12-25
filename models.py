from keras import backend
import matplotlib.pylab as plt
import numpy as np
import os
import pickle
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Reshape, PReLU, Conv2D, Concatenate, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Add, Activation, ReLU, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence



# https://github.com/rajatkb/RDNSR-Residual-Dense-Network-for-Super-Resolution-Keras

def RDBlocks(x, name, count=6, g=32):
    ## 6 layers of RDB block
    ## this thing need to be in a damn loop for more customisability
        li = [x]
        pas = Conv2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu' , name = name + '_conv1')(x)

        for i in range(2 , count + 1):
            li.append(pas)
            out =  Concatenate(axis=3)(li) # conctenated out put
            pas = Conv2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name = name + '_conv' + str(i))(out)

        # feature extractor from the dense net
        li.append(pas)
        out = Concatenate(axis=3)(li)
        feat = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same', activation='relu' , name = name + '_Local_Conv')(out)

        feat = Add()([feat, x])
        return feat

def RDN(inputs, RDB_count=20):
    pass1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
    pass2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pass1)

    RDB = RDBlocks(pass2, 'RDB1')
    RDBlocks_list = [RDB,]
    for i in range(2, RDB_count + 1):
        RDB = RDBlocks(RDB,'RDB' + str(i))
        RDBlocks_list.append(RDB)
    out = Concatenate(axis=3)(RDBlocks_list)
    out = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(out)

    output = Add()([out, pass1])

    output = Conv2D(filters=1, kernel_size=(3,3), strides=(1, 1), padding='same')(output)

    return output





# https://github.com/nikhilroxtomar/Deep-Residual-Unet

def bn_act(x, act=True):
    x = BatchNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c

def ResUNet(inputs):
    f = [16, 32, 64, 128, 256]
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    return outputs






class ReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = backend.get_value(self.model.optimizer.lr)
        logs["wd"] = self.model.optimizer.weight_decay
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Learning rate reduction is conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = backend.get_value(self.model.optimizer.lr)
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        backend.set_value(self.model.optimizer.lr, new_lr)
                        if new_lr != old_lr:
                            old_wd = self.model.optimizer.weight_decay
                            new_wd = old_wd * self.factor
                            self.model.optimizer.weight_decay = new_wd
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nEpoch {epoch +1}: "
                                "ReduceLROnPlateau reducing "
                                f"learning rate to {new_lr}."
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0



def getModel(name):
    inputs = Input((128, 128, 1))

    if name == 'rdn':
        output = RDN(inputs, RDB_count=4)
    elif name == 'resunet':
        output = ResUNet(inputs)
    else:
        # Dummy model
        output = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)

    model = Model(inputs=inputs, outputs=output)

    return model
