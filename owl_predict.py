from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

model =  load_model('/home/vpanchbh/Downloads/owl_model.h5')
test_dir = './test/'

owl_datagen = ImageDataGenerator(rescale=1./255)
no_owl_datagen = ImageDataGenerator(rescale=1./255)

owl_generator = owl_datagen.flow_from_directory(
        test_dir,
        target_size=(640, 480),
        batch_size=16,
	class_mode='binary')

#prediction = model.predict_generator(owl_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
#print(prediction)

path = test_dir+'owl'

for i in os.listdir(path):
    x = load_img(path+'/'+i, target_size=(640,480))
    x = img_to_array(x)
    x = x.reshape((1,) + x.shape)
    x = x/255.
    val = model.predict(x)
    if val < 0.5:
	print(i,val)
