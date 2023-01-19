# Librerías y Rutas
from tensorflow import keras
from PIL import Image, ImageFilter, ImageChops
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf


SIZE_INPUT = 128

# Modelos y Capas
def build_model():
    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(kernel_size=(3,3), input_shape=(128,128,3), 
                                  filters=3, activation="relu", padding='same'))
    
    model.add(keras.layers.Conv2D(kernel_size=(2,2), filters=32, activation="relu", 
                                  padding='same', kernel_regularizer=keras.regularizers.l2(0.01)))
    
    model.add(keras.layers.Conv2D(kernel_size=(3,3), filters=64, activation="relu", 
                                 padding='same', kernel_regularizer=keras.regularizers.l2(0.02)))
    
    model.add(keras.layers.Conv2D(kernel_size=(1,1), filters=64, activation="relu", 
                                 padding='same'))
    
    model.add(keras.layers.Conv2D(kernel_size=(1,1), filters=128, activation="relu", 
                                 padding='same'))
     
    model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=64, activation="relu", 
                                 padding='same'))
      
    model.add(keras.layers.Conv2DTranspose(kernel_size=(3,3), filters=64, activation="relu", 
                                 padding='same', 
                                           kernel_regularizer=keras.regularizers.l2(0.01)))
    
    model.add(keras.layers.Conv2DTranspose(kernel_size=(2,2), filters=32, activation="relu", 
                                 padding='same'))
    
    
    model.add(keras.layers.Conv2DTranspose(kernel_size=(3,3), filters=3, activation="relu", 
                                 padding='same'))
        
    model.compile(keras.optimizers.Adamax(), 
                  loss='mae', metrics=['acc', 'mse'])
  
    return model

model = build_model()
model.summary()




def make_image(array, width, height):
    newIm = Image.new('RGB', (128*width, 128*height), color=(0,0,0))
    
    for i in range(height):
        for j in range(width):
            im = Image.fromarray(array[i*width+j].astype('uint8'))
            newIm.paste(im, (j*128,i*128))
    
    return newIm
    

