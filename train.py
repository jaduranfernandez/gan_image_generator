import numpy as np
from models.face_model import build_GAN, build_generator, build_discriminator


x_train = np.load("data/faces_4000.npy")

batch_size = 32
n_iterations = 1000
n_bathes = x_train.shape[0]/batch_size
input_size = 100


generator = build_generator()
discriminator = build_discriminator()
gan = build_GAN(generator, discriminator)

for i in range(1,n_iterations+1):
    print("Epoch " + str(i))

    # Crear un "batch" de imágenes falsas y otro con imágenes reales
    noise = np.random.normal(0,1,[batch_size,input_size])
    false_images = generator.predict(noise)

    idx = np.random.randint(low=0, high=x_train.shape[0],size=batch_size)
    true_images = x_train[idx]

    discriminator.trainable = True
    dError_reales = discriminator.train_on_batch(true_images,
        np.ones(batch_size)*0.9)
    dError_falsas = discriminator.train_on_batch(false_images,
        np.zeros(batch_size)*0.1)
    
    discriminator.trainable = False
    noise = np.random.normal(0,1,[batch_size,input_size])
    gError = gan.train_on_batch(noise, np.ones(batch_size))
    if i==1 or i%100 == 0:
        generator.save(str.format("saved_models/generator_{0}.h5",i))

