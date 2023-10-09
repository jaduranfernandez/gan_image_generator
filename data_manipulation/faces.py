import matplotlib.pyplot as plt 
import numpy as np
from keras import layers, Sequential


def create_dataset(folder_path, file_name, data_path, first_image, last_image):
    
    dimensions = [128,128,3]
    values = np.zeros((last_image - first_image+1, dimensions[0],dimensions[1],dimensions[2]))
    # Resizing and rescaling
    resize_and_scale = Sequential()
    resize_and_scale.add(layers.Resizing(dimensions[0], dimensions[1]))
    resize_and_scale.add(layers.Rescaling(1./255))


    for it in range(first_image, last_image+1):
        filename = str(it).zfill(6) + ".jpg"
        image = plt.imread(data_path + filename)
        new_image = resize_and_scale(image)
        new_image = (2*new_image) - 1
        values[it-1,:] = new_image
    np.save(folder_path + file_name, values)

