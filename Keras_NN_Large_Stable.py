
# coding: utf-8

# In[1]:


import keras
import numpy as np
from matplotlib import pyplot as plt
import math
import pandas
from keras import Sequential
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
import keras.callbacks as callbacks
from keras.utils import multi_gpu_model


# input1 = Missing image (Masked by cloud)
# 
# input2 = Referenced images (cloud-free)

# ### Implement Model

# In[2]:


input1 = Input(shape=(400,400,1), name='input1')
input2 = Input(shape=(400,400,1), name='input2')
input3 = Input(shape=(400,400,1), name='input3')

conv_1 = Conv2D(30, (3, 3), padding='same')(input1)
conv_2 = Conv2D(30, (3, 3), padding='same')(input2)
conv_3 = Conv2D(60, (3, 3), padding='same')(input3)

concat_1_2 = concatenate([conv_1, conv_2], axis=-1)
concat_1_2 = Activation('relu')(concat_1_2)

concat_1_2 = Dropout(0.3)(concat_1_2)

feature_3 = Conv2D(filters=20, kernel_size=(9, 9),padding='same')(concat_1_2)
feature_5 = Conv2D(filters=20, kernel_size=(5, 5),padding='same')(concat_1_2)
feature_7 = Conv2D(filters=20, kernel_size=(7, 7),padding='same')(concat_1_2)
feature_3_5_7 = concatenate([feature_3, feature_5, feature_7])
feature_3_5_7 = Activation('relu')(feature_3_5_7)

# feature_3_5_7 = Dropout(0.3)(feature_3_5_7)

sum0 = add([concat_1_2, feature_3_5_7])

conv1 = Conv2D(filters=60, kernel_size=(3,3), padding='same',activation='relu')(sum0)
conv2 = Conv2D(filters=60, kernel_size=(3,3), padding='same',activation='relu')(conv1)

sum1 = add([conv2, conv_3])
conv3 = Conv2D(filters=60, kernel_size=(3,3), dilation_rate=2, padding='same',activation='relu')(sum1)
conv4 = Conv2D(filters=60, kernel_size=(3,3), dilation_rate=3, padding='same',activation='relu')(conv3)
conv5 = Conv2D(filters=60, kernel_size=(3,3), dilation_rate=2, padding='same',activation='relu')(conv4)
sum2 = add([conv5, conv_3])

conv6 = Conv2D(filters=60, kernel_size=(3,3), padding='same',activation='relu')(sum2)

conv6 = Dropout(0.3)(conv6)

conv6_2 = add([conv6, conv1])

conv7 = Conv2D(filters=1, kernel_size=(3,3), padding='same')(conv6_2)

model = Model([input1, input2, input3], conv7)

model = multi_gpu_model(model, gpus=2, cpu_merge=True)

import h5py, sys

ratio = 65535.
f = h5py.File(sys.argv[1], 'r')
label = f['label'][:].astype('float32') / ratio
masked = f['masked'][:].astype('float32') / ratio
ref = f['ref'][:].astype('float32') / ratio
mask = f['mask'][:].astype('float32') / ratio
f.close()

nTrain = int(0.8 * label.shape[0])

trainY = label[:nTrain]
trainX_1 = masked[:nTrain]
trainX_2 = ref[:nTrain]
trainX_3 = mask[:nTrain]

testY = label[nTrain:]
testX_1 = masked[nTrain:]
testX_2 = ref[nTrain:]
testX_3 = mask[nTrain:]
f.close()

img_rows, img_cols = 400, 400
out_rows, out_cols = 400, 400

trainX_3 = trainX_1 + np.multiply(trainX_2, np.reshape(trainX_3, (trainX_3.shape[0], img_rows, img_cols, 1)))
testX_3 = testX_1 + np.multiply(testX_2, np.reshape(testX_3, (testX_3.shape[0], img_rows, img_cols, 1)))


def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.3
    epochs_drop = 50
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# In[5]:


adam = Adam(lr=0.0001)

model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])

lrate = LearningRateScheduler(step_decay)
callback_list = [callbacks.ModelCheckpoint('checkpoint_' + sys.argv[2], monitor='val_PSNRLoss', save_best_only=True,
                                           mode='max', save_weights_only=True, verbose=0)]

batch_size = 32
nb_epoch = 200

history = model.fit([trainX_1,trainX_2,trainX_3], trainY, batch_size=batch_size, epochs=nb_epoch, callbacks=callback_list,
                    verbose=1, validation_data=([testX_1, testX_2, testX_3], testY), verbose=1)  

model.save_weights(sys.argv[2])

