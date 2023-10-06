import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
from keras import layers, Sequential, optimizers

def build_generator():
    model = Sequential()
    model.add(layers.Dense(1024*4*4, input_shape=(100,)))
    model.add(layers.BatchNormalization(momentum=0.3))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((4,4,1024))) # 4x4x1024

    model.add(layers.Conv2DTranspose(512,(5,5), strides=(2,2), padding='same')) # 8x8x512
    model.add(layers.BatchNormalization(momentum=0.3))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(256,(5,5), strides=(2,2), padding='same')) # 16x16x256
    model.add(layers.BatchNormalization(momentum=0.3))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(128,(5,5), strides=(2,2), padding='same')) # 32x32x128
    model.add(layers.BatchNormalization(momentum=0.3))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(64,(5,5), strides=(2,2), padding='same')) # 64x64x64
    model.add(layers.BatchNormalization(momentum=0.3))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(3,(5,5), strides=(2,2), padding='same')) # 128x128x3 --> desired output
    model.add(layers.Activation("tanh"))
    adam = optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(optimizer=adam, loss="binary_crossentropy")
    return model




def build_discriminator():
    model = Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding="same", input_shape=(128,128,3,))) # 64x64x64
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(128,(5,5), strides=(2,2), padding='same')) # 32x32x128
    model.add(layers.BatchNormalization(momentum=0.3))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(256,(5,5), strides=(2,2), padding='same')) # 16x16x256
    model.add(layers.BatchNormalization(momentum=0.3))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(512,(5,5), strides=(2,2), padding='same')) # 8x8x512
    model.add(layers.BatchNormalization(momentum=0.3))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(1024,(5,5), strides=(2,2), padding='same')) # 4x4x1024
    model.add(layers.BatchNormalization(momentum=0.3))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    
    adam = optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(optimizer=adam, loss="binary_crossentropy")
    
    return model



def build_GAN(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    adam = optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(optimizer=adam, loss="binary_crossentropy")
    return model



# model = build_discriminator()
# model.trainable = False
# x = np.random.normal(0,1,(1,128,128,3))
# print(x.shape)
# y = model(x)
# print(y.shape)