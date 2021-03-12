import os, sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]

import cv2
import numpy as np
from keras import layers
from keras import models
from keras.models import Sequential, load_model
from keras import optimizers
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils.np_utils import to_categorical
from keras.layers import  Conv2D,MaxPooling2D,Activation,Flatten,Dense, Dropout, BatchNormalization
from keras.applications import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,load_img
import matplotlib.pyplot as plt
from keras import callbacks

def data_preprocess(train_dir, val_dir):

	print('Number of images in creamy_paste: ', len(os.listdir(os.path.join(train_dir, 'creamy_paste'))), len(os.listdir(os.path.join(val_dir, 'creamy_paste'))))
	print('Number of images in diced: ', len(os.listdir(os.path.join(train_dir, 'diced'))), len(os.listdir(os.path.join(val_dir, 'diced'))))
	print('Number of images in floured: ', len(os.listdir(os.path.join(train_dir, 'floured'))), len(os.listdir(os.path.join(val_dir, 'floured'))))
	print('Number of images in grated: ', len(os.listdir(os.path.join(train_dir, 'grated'))), len(os.listdir(os.path.join(val_dir, 'grated'))))
	print('Number of images in juiced: ', len(os.listdir(os.path.join(train_dir, 'juiced'))), len(os.listdir(os.path.join(val_dir, 'juiced'))))
	print('Number of images in jullienne: ', len(os.listdir(os.path.join(train_dir, 'jullienne'))), len(os.listdir(os.path.join(val_dir, 'jullienne'))))
	print('Number of images in mixed: ', len(os.listdir(os.path.join(train_dir, 'mixed'))), len(os.listdir(os.path.join(val_dir, 'mixed'))))
	print('Number of images in other: ', len(os.listdir(os.path.join(train_dir, 'other'))), len(os.listdir(os.path.join(val_dir, 'other'))))
	print('Number of images in peeled: ', len(os.listdir(os.path.join(train_dir, 'peeled'))), len(os.listdir(os.path.join(val_dir, 'peeled'))))
	print('Number of images in sliced: ', len(os.listdir(os.path.join(train_dir, 'sliced'))), len(os.listdir(os.path.join(val_dir, 'sliced'))))
	print('Number of images in whole: ', len(os.listdir(os.path.join(train_dir, 'whole'))), len(os.listdir(os.path.join(val_dir, 'whole'))))

	train_datagen = ImageDataGenerator(
      		samplewise_center=True,
      		samplewise_std_normalization=True,
			rescale=1./255,
			rotation_range=40,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode='nearest')

	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
			train_dir,
			target_size=(150, 150),
			batch_size=64)

	valid_generator = test_datagen.flow_from_directory(
	        val_dir,
	        target_size=(150, 150),
	        batch_size=64)

	return train_generator, valid_generator


def build_model(train_generator, valid_generator):

	# callbacks
	model_path = 'cooking_state_best_model.h5'
	checkpoint = callbacks.ModelCheckpoint(
	        filepath=model_path, 
	        monitor='val_acc', 
	        verbose=1, 
	        save_best_only=True)
	callbacks = [checkpoint]

	# model architecture
	model = models.Sequential()
	model.add(layers.Conv2D(64, (3, 3), input_shape=(150, 150, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D((2, 2)))

	model.add(layers.Conv2D(64, (3, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D((2, 2)))

	model.add(Dropout(0.15))

	model.add(layers.Conv2D(128, (3, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D((2, 2)))

	model.add(layers.Conv2D(128, (3, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D((2, 2)))

	model.add(Dropout(0.5))

	model.add(layers.Conv2D(128, (3, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.BatchNormalization())
	# model.add(layers.MaxPooling2D((2, 2)))

	model.add(layers.Conv2D(128, (3, 3)))
	model.add(layers.Activation('relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D((2, 2)))

	model.add(layers.Flatten())

	model.add(layers.Dense(64))
	model.add(layers.Activation('relu'))

	model.add(layers.Dense(11))
	model.add(layers.Activation('softmax'))

	# print(model.summary())

	# compiling the model
	model.compile(loss='categorical_crossentropy',
	              optimizer=optimizers.Nadam(lr= 0.01),
	              metrics=['acc'])

	# fit the model on train_data
	history = model.fit(
	      train_generator,
	      steps_per_epoch=100,
	      epochs=50,
	      validation_data=valid_generator,
	      callbacks = callbacks)

	# loading the best model and checking validation accuracy
	model = models.load_model(model_path)
	val_loss, val_accuracy = model.evaluate_generator(valid_generator)
	print(val_accuracy)

	# getting training and validation data performance
	train_acc = history.history['acc']
	val_acc = history.history['val_acc']
	train_loss = history.history['loss']
	val_loss = history.history['val_loss']

	return train_acc, val_acc, train_loss, val_loss


def plot(train_acc, val_acc, train_loss, val_loss):

	epochs = range(len(train_))
	plt.plot(epochs, train_acc, 'r', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, train_loss, 'r', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

def main():

	train_dir = 'C:/Users/Dell/Downloads/train (1)/train'
	val_dir = 'C:/Users/Dell/Downloads/valid (1)/valid'

	# passing the data through ImageDataGenerator
	train_generator, valid_generator = data_preprocess(train_dir, val_dir)

	# building the model and getting its performance on train and validation data
	train_acc, val_acc, train_loss, val_loss = build_model(train_generator, valid_generator)

	# plotting the accuracy and loss graphs
	plot(train_acc, val_acc, train_loss, val_loss)

if __name__ == '__main__':
	main()