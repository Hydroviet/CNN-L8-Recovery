
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

input1 = Input(shape=(400,400,1), name='input1')
input2 = Input(shape=(nInput, 400,400,1), name='input2')

time1 = TimeDistributed(Conv2D(30, (3, 3), padding='same', activation='relu'))(input2)
time2 = TimeDistributed(Conv2D(30, (5, 5), padding='same', activation='relu'))(time1)
time3 = TimeDistributed(Conv2D(30, (7, 7), padding='same', activation='relu'))(time2)
time4 = Dropout(0.3)(time3)
time = TimeDistributed(Dense(1))(time4)
time = Lambda(sliced)(time)

conv_1 = Conv2D(30, (3, 3), padding='same')(input1)
conv_2 = Conv2D(30, (3, 3), padding='same')(time)

concat_1_2 = concatenate([conv_1, conv_2], axis=-1)
concat_1_2 = Activation('relu')(concat_1_2)

concat_1_2 = Dropout(0.3)(concat_1_2)

feature_3 = Conv2D(filters=20, kernel_size=(3, 5),padding='same')(concat_1_2)
feature_5 = Conv2D(filters=20, kernel_size=(5, 5),padding='same')(concat_1_2)
feature_7 = Conv2D(filters=20, kernel_size=(7, 7),padding='same')(concat_1_2)
feature_3_5_7 = concatenate([feature_3, feature_5, feature_7])
feature_3_5_7 = Activation('relu')(feature_3_5_7)

feature_3_5_7 = Dropout(0.3)(feature_3_5_7)

sum0 = add([concat_1_2, feature_3_5_7])

conv1 = Conv2D(filters=60, kernel_size=(3,3), padding='same',activation='relu')(sum0)
conv2 = Conv2D(filters=30, kernel_size=(3,3), padding='same',activation='relu')(conv1)

sum1 = add([conv2, conv_2])
conv3 = Conv2D(filters=60, kernel_size=(3,3), dilation_rate=2, padding='same',activation='relu')(sum1)
conv4 = Conv2D(filters=60, kernel_size=(3,3), dilation_rate=3, padding='same',activation='relu')(conv3)
conv5 = Conv2D(filters=60, kernel_size=(3,3), dilation_rate=2, padding='same',activation='relu')(conv4)
sum2 = add([conv5, conv3])

conv6 = Conv2D(filters=60, kernel_size=(3,3), padding='same',activation='relu')(sum2)

conv7 = Conv2D(filters=1, kernel_size=(3,3), padding='same')(conv6)


model = Model([input1, input2], conv7)

# model = multi_gpu_model(model, gpus=2)


# ## Read dataset

# In[3]:


import h5py

f = h5py.File('dataset5.hdf5', 'r')

in1 = f['in1'][:]
in2 = f['in2'][:]
in3 = f['in3'][:]
in4 = f['in4'][:]
in5 = f['in5'][:]
masked = f['masked'][:]
out = f['out'][:]
f.close()


# ### Config Model

# In[7]:


def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.3
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# In[14]:


nTrain = int(0.7 * len(in1))
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


adam = Adam(lr=0.0001)

model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])

lrate = LearningRateScheduler(step_decay)
callback_list = [callbacks.ModelCheckpoint('cont_1608_adam.h5', monitor='val_PSNRLoss', save_best_only=True,
                                           mode='max', save_weights_only=True, verbose=1)]

batch_size = 1
nb_epoch = 20

history = model.fit([trainX_1, trainX_2], trainY, batch_size=batch_size, epochs=nb_epoch, callbacks=callback_list,
                    verbose=1, validation_data=([testX_1, testX_2], testY))  


# In[ ]:


y_pred = model.predict([testX_1[10:11], testX_2[10:11], testX_3[10:11]])


# Original Input

# In[ ]:


print(history.history.keys())


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('MSE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'][:10])
plt.plot(history.history['val_loss'][:10])
plt.title('MSE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# summarize history for loss
plt.plot(history.history['PSNRLoss'])
plt.plot(history.history['val_PSNRLoss'])
plt.title('PNSR loss')
plt.ylabel('PSNR')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


model.save_weights('model1608_cont.h5')

