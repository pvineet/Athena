
# coding: utf-8

# In[10]:


# coding: utf-8

import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, Sequential


class VGG16_Model:
# train_layers -  number of layers to be trained
# fc_layer_size - Dense layer size
    def __init__(self, train_layers=0, fc_layer_size=128):
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(640,480,3))
        for i in range(len(vgg.layers)):
            if i < len(vgg.layers) - train_layers:
                vgg.layers[i].trainable = False
            print(vgg.layers[i].name, vgg.layers[i].trainable)

        # Create the model
        self.model = Sequential()

        # Add the vgg convolutional base model
        self.model.add(vgg)
        # Add new layers
        self.model.add(Flatten())
        self.model.add(Dense(fc_layer_size, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()


owl_model = VGG16_Model(train_layers=2)

owl_model.model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
#image_dir = '/Volumes/Seagate Slim Drive/Vineet/Owl_dataset_resized'
image_dir = '.'
train_generator = train_datagen.flow_from_directory(
        image_dir+'/train',
        target_size=(640, 480),
        batch_size=16,
        class_mode='binary')

val_generator = val_datagen.flow_from_directory(
        image_dir+'/val',
        target_size=(640, 480),
        batch_size=16,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        image_dir+'/test',
        target_size=(640, 480),
        batch_size=32,
        class_mode='binary')

owl_model.model.fit_generator(
        train_generator,
        epochs=20,
        workers=8,
        validation_data=val_generator,
    	validation_steps=50)


save_dir = "/home/vpanchbh/Downloads/"
model_name = "owl_model.h5"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
owl_model.model.save(model_path)
print('Saved trained model at %s ' % model_path)

x,y = owl_model.model.evaluate_generator(test_generator)
print("Test loss {}".format(x))
print("Test accuracy {}".format(y))
