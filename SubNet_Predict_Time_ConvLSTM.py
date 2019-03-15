
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
from keras.layers.wrappers import TimeDistributed
from keras.utils import multi_gpu_model


# input1 = Missing image (Masked by cloud)
# 
# input2 = Referenced images (cloud-free)

# ### Implement Model

# In[2]:


def sliced(x):
    return x[:,0,:,:]

nInput = 5

input2 = Input(shape=(nInput, 400,400,1), name='input2')

time1 = ConvLSTM2D(64, (9, 9), padding='same', activation='relu', return_sequences=True)(input2)
time2 = ConvLSTM2D(32, (3, 3), padding='same', activation='relu', return_sequences=True)(time1)
time2 = ConvLSTM2D(16, (7, 7), padding='same', activation='relu', return_sequences=True)(time2)
time3 = ConvLSTM2D(1, (5, 5), padding='same', activation='relu')(time2)
time4 = Dropout(0.3)(time3)

model = Model(input2, time4)

model = multi_gpu_model(model, gpus=2)


# ## Read dataset

# In[3]:


import h5py

f = h5py.File('dataset5.hdf5', 'r')

in1 = f['in1'][:].astype('float32')
in2 = f['in2'][:].astype('float32')
in3 = f['in3'][:].astype('float32')
in4 = f['in4'][:].astype('float32')
in5 = f['in5'][:].astype('float32')
masked = f['masked'][:].astype('float32')
out = f['out'][:].astype('float32')
f.close()

# scale
ratio = in1.max()
ratio = max(ratio, in2.max())
ratio = max(ratio, in3.max())
ratio = max(ratio, in4.max())
ratio = max(ratio, in5.max())
ratio = max(ratio, masked.max())
ratio = max(ratio, out.max())

in1 /= ratio
in2 /= ratio
in3 /= ratio
in4 /= ratio
in5 /= ratio
masked /= ratio
out /= ratio


# ### Config Model

# In[7]:


def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.3
    epochs_drop = 50
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# In[14]:


nTrain = int(0.9 * len(in1))
nTest = len(in1) - nTrain

trainX_2 = np.zeros((nTrain, 5, 400, 400, 1), dtype='uint8')
testX_2 = np.zeros((nTest, 5, 400, 400, 1), dtype='uint8')

trainX_1 = masked[:nTrain,:,:,:]
trainX_2[:,0,:,:,0] = in1[:nTrain,:,:,0]
trainX_2[:,1,:,:,0] = in2[:nTrain,:,:,0]
trainX_2[:,2,:,:,0] = in3[:nTrain,:,:,0]
trainX_2[:,3,:,:,0] = in4[:nTrain,:,:,0]
trainX_2[:,4,:,:,0] = in5[:nTrain,:,:,0]
trainY = out[:nTrain,:,:,:]

testX_1 = masked[nTrain:,:,:,:]
testX_2[:,0,:,:,0] = in1[nTrain:,:,:,0]
testX_2[:,1,:,:,0] = in2[nTrain:,:,:,0]
testX_2[:,2,:,:,0] = in3[nTrain:,:,:,0]
testX_2[:,3,:,:,0] = in4[nTrain:,:,:,0]
testX_2[:,4,:,:,0] = in5[nTrain:,:,:,0]
testY = out[nTrain:,:,:,:]


# In[15]:
import sys

adam = Adam(lr=0.0001)

model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])

lrate = LearningRateScheduler(step_decay)
callback_list = [callbacks.ModelCheckpoint('cp_' + sys.argv[3], monitor='val_PSNRLoss', save_best_only=True,
                                           mode='max', save_weights_only=True, verbose=1)]
batch_size = int(sys.argv[1])
nb_epoch = int(sys.argv[2])

history = model.fit(trainX_2, trainY, batch_size=batch_size, epochs=nb_epoch, callbacks=callback_list,
                    verbose=1, validation_data=(testX_2[:-2], testY[:-2]))  

model.save_weights(sys.argv[3])

