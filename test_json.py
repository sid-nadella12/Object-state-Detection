import os, cv2
import sys
import numpy as np
from keras import layers
from keras import models
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import  Conv2D,MaxPooling2D,Activation,Flatten,Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,load_img
import matplotlib.pyplot as plt
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]


def generate_json(test_path):
	# classes
	classes=['creamy_paste','diced','floured','grated','juiced','jullienne','mixed','other','peeled','sliced','whole']

	# loading the model
	model_path = 'cooking_state_best_model.h5'
	model = models.load_model(model_path)
	data = {}

	# loading images and predicting the classes
	for images in os.listdir(test_path):

		image = load_img(os.path.join(test_path, images))
		image = np.asarray(image)
		image = cv2.resize(image, (150, 150))
		image = np.expand_dims(image, axis=0)
		image = image * 1.0 / 255

		result = model.predict(image)
		index_of_maximum = np.where(result == np.max(result))
		class_index = int(index_of_maximum[1])
		class_name = classes[class_index]

		filename = os.path.split(test_path)[1] +'/'+ images
		data[filename] = class_name
		json_data = json.dumps(data)

	return json_data

def main():
	
 	# give the test folder path
	test_path = 'C:/Users/Dell/Downloads/test_anonymous/test/test_anonymous'

	# generate json data
	json_data = generate_json(test_path)

	# write the jjson data into the file
	with open("sudheer_result_model.json", "w") as outfile: 
		outfile.write(json_data) 
	print('json file generated')


if __name__ == '__main__':
	main()