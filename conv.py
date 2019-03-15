import keras
import numpy as np
import math
from keras import Sequential
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
import keras.callbacks as callbacks
from keras.layers.wrappers import TimeDistributed
from keras.utils import multi_gpu_model
from matplotlib import pyplot as plt

def multiply(ip):
	raw = ip[0].set_shape([1, 400, 400, 1])
	ref = ip[1].set_shape([1, 400, 400, 1])
	shapeInfo = [1, 400, 400, 1]	
	kzeros = K.zeros(shape=shapeInfo)
	mask = K.equal(raw, kzeros)
	ret = raw + ref
	return ret
		
nInput = 5
n_pixel = 30

input1 = Input(shape=(400,400,1), name='input1')
input2 = Input(shape=(nInput, 400,400,1), name='input2')

pred = ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),padding='same', return_sequences=True)(input2)
pred = BatchNormalization()(pred)
pred = ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),padding='same', return_sequences=True)(pred)
pred = BatchNormalization()(pred)
pred = ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),padding='same', return_sequences=True)(pred)
pred = BatchNormalization()(pred)
pred = ConvLSTM2D(filters=n_pixel, kernel_size=(3, 3),padding='same', return_sequences=False)(pred)
pred = BatchNormalization()(pred)
pred = Conv2D(filters=1, kernel_size=(3, 3),padding='same', activation='sigmoid')(pred)

pred = Dense(1)(pred)
pred = Dropout(0.3)(pred)

# input3 = Lambda(multiply)([input1, pred])

conv_1 = Conv2D(30, (3, 3), padding='same')(input1)
conv_2 = Conv2D(30, (3, 3), padding='same')(pred)
conv_3 = Conv2D(60, (3, 3), padding='same')(pred)

concat_1_2 = concatenate([conv_1, conv_2], axis=-1)
concat_1_2 = Activation('relu')(concat_1_2)

concat_1_2 = Dropout(0.3)(concat_1_2)

feature_3 = Conv2D(filters=20, kernel_size=(9, 9),padding='same')(concat_1_2)
feature_5 = Conv2D(filters=20, kernel_size=(5, 5),padding='same')(concat_1_2)
feature_7 = Conv2D(filters=20, kernel_size=(7, 7),padding='same')(concat_1_2)
feature_3_5_7 = concatenate([feature_3, feature_5, feature_7])
feature_3_5_7 = Activation('relu')(feature_3_5_7)

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
model = Model([input1, input2], conv7)

model = multi_gpu_model(model, gpus=2)

import h5py


f = h5py.File('test.hdf5', 'r')
in1 = f['in1'][:].astype('float32')
in2 = f['in2'][:].astype('float32')
in3 = f['in3'][:].astype('float32')
in4 = f['in4'][:].astype('float32')
in5 = f['in5'][:].astype('float32')
masked = f['masked'][:].astype('float32')
out = f['out'][:].astype('float32')
f.close()

ratio = 65535.
in1 /= ratio
in2 /= ratio
in3 /= ratio
in4 /= ratio
in5 /= ratio
masked /= ratio
out /= ratio

def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 50
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


# In[14]:


X2 = np.zeros((9, 5, 400, 400, 1), dtype='float32')
X1 = masked
X2[:,0,:,:,0] = in1[:,:,:,0]
X2[:,1,:,:,0] = in2[:,:,:,0]
X2[:,2,:,:,0] = in3[:,:,:,0]
X2[:,3,:,:,0] = in4[:,:,:,0]
X2[:,4,:,:,0] = in5[:,:,:,0]
Y = out[:,:,:,:]

adam = Adam(lr=0.0001)

model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
model.load_weights('model_conv_2808.h5')

rgb_out = np.zeros((400, 400, 3), dtype=X1.dtype)
rgb_in = np.zeros((400, 400, 3), dtype=X1.dtype)
rgb_gt = np.zeros((400, 400, 3), dtype=X1.dtype)
k = -1
for i in range(3):
	k += 1
	plt.imsave('simple_conv_' + + str(i) + '_gt.png', Y[k,:,:,0] * 255)
	for j in range(5):
		plt.imsave('simple_conv_' + str(i) + '_ref_' + str(j) + '.png', X2[k,j,:,:,0] * 255)
	print('Simple Conv:')
	print(model.evaluate([trainX_1[k:k+1], trainX_2[k:k+1]], trainY[k:k+1]))
	output = model.predict([trainX_1[k:k+1],trainX_2[k:k+1]])
	rgn_out[:,:,i] = output[0,:,:,0]
	rgb_in[:,:,i] = X1[k,:,:,0]
	rgb_gt[:,:,i] = Y[k,:,:,0]
	plt.imsave('simple_conv_' + str(i) + '_in.png', X1[k,:,:,0] * 255)
	plt.imsave('simple_conv_' + str(i) + '_out.png', output[0,:,:,0] * 255)

plt.imsave('simple_conv_in.png', rgb_in * 255)
plt.imsave('simple_conv_out.png', rgb_out * 255)
plt.imsave('simple_conv_gt.png', rgb_gt * 255)

rgb_out = np.zeros((400, 400, 3), dtype=X1.dtype)
rgb_in = np.zeros((400, 400, 3), dtype=X1.dtype)
rgb_gt = np.zeros((400, 400, 3), dtype=X1.dtype)
for i in range(3):
	k += 1
	plt.imsave('med_conv_' + + str(i) + '_gt.png', Y[k,:,:,0] * 255)
	for j in range(5):
		plt.imsave('med_conv_' + str(i) + '_ref_' + str(j) + '.png', X2[k,j,:,:,0] * 255)
	print('Med Conv:')
	print(model.evaluate([trainX_1[k:k+1], trainX_2[k:k+1]], trainY[k:k+1]))
	output = model.predict([trainX_1[k:k+1],trainX_2[k:k+1]])
	rgn_out[:,:,i] = output[0,:,:,0]
	rgb_in[:,:,i] = X1[k,:,:,0]
	rgb_gt[:,:,i] = Y[k,:,:,0]
	plt.imsave('med_conv_' + str(i) + '_in.png', X1[k,:,:,0] * 255)
	plt.imsave('med_conv_' + str(i) + '_out.png', output[0,:,:,0] * 255)

plt.imsave('med_conv_in.png', rgb_in * 255)
plt.imsave('med_conv_out.png', rgb_out * 255)
plt.imsave('med_conv_gt.png', rgb_gt * 255)

rgb_out = np.zeros((400, 400, 3), dtype=X1.dtype)
rgb_in = np.zeros((400, 400, 3), dtype=X1.dtype)
rgb_gt = np.zeros((400, 400, 3), dtype=X1.dtype)
for i in range(3):
	k += 1
	plt.imsave('complex_conv_' + + str(i) + '_gt.png', Y[k,:,:,0] * 255)
	for j in range(5):
		plt.imsave('complex_conv_' + str(i) + '_ref_' + str(j) + '.png', X2[k,j,:,:,0] * 255)
	print('complex Conv:')
	print(model.evaluate([trainX_1[k:k+1], trainX_2[k:k+1]], trainY[k:k+1]))
	output = model.predict([trainX_1[k:k+1],trainX_2[k:k+1]])
	rgn_out[:,:,i] = output[0,:,:,0]
	rgb_in[:,:,i] = X1[k,:,:,0]
	rgb_gt[:,:,i] = Y[k,:,:,0]
	plt.imsave('complex_conv_' + str(i) + '_in.png', X1[k,:,:,0] * 255)
	plt.imsave('complex_conv_' + str(i) + '_out.png', output[0,:,:,0] * 255)

plt.imsave('complex_conv_in.png', rgb_in * 255)
plt.imsave('complex_conv_out.png', rgb_out * 255)
plt.imsave('complex_conv_gt.png', rgb_gt * 255)