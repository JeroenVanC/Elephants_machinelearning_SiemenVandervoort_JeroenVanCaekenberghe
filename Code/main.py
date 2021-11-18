# Import
from PIL import Image
import imageio as iio
import numpy as np
import random

# Constants
newSize = 50
train_images = 510

def cropimages():
	print("Cropping images ...")
	for number in range(1, 421):
		print("Opening Af tr image: " + str(number))
		im = Image.open("Dataset/Train/African/af_tr" + str(number) + ".jpg")

		resized_image = im.resize((newSize, newSize))
		resized_image.convert('RGB').save("Dataset/Train/Resized_Images/African/African_" + str(number) + ".jpg")

	for number in range(1, 91):
		print("Opening Af te image: " + str(number))
		im = Image.open("Dataset/Train/African/Af_te" + str(number) + ".jpg")

		resized_image = im.resize((newSize, newSize))
		resized_image.convert('RGB').save("Dataset/Train/Resized_Images/African/African_" + str(number + 420) + ".jpg")

	for number in range(1, 421):
		print("Opening As tr image: " + str(number))
		im = Image.open("Dataset/Train/Asian/As_tr" + str(number) + ".jpg")

		resized_image = im.resize((newSize, newSize))
		resized_image.convert('RGB').save("Dataset/Train/Resized_Images/Asian/Asian_" + str(number) + ".jpg")

	for number in range(1, 91):
		print("Opening As te image: " + str(number))
		im = Image.open("Dataset/Train/Asian/As_te" + str(number) + ".jpg")

		resized_image = im.resize((newSize, newSize))
		resized_image.convert('RGB').save("Dataset/Train/Resized_Images/Asian/Asian_" + str(number + 420) + ".jpg")

	print("Images cropped!");

def loadimages():
	print("Loading images ...")
	train_set_x = []
	train_set_y = []
	for i in range(0, 2 * train_images):
		train_set_x.append(0)
		train_set_y.append(0)

	#Generate list with random numbers
	randomList = []
	randomList = random.sample(range(0, (2 * train_images)), 2 * train_images)

	#Use a counter to iterate through the randomList
	counter = 0

	# Load African images
	for number in range(1, train_images + 1):
		Af_rgb = iio.imread("Dataset/Train/Resized_Images/African/African_" + str(number) + ".jpg")
		train_set_x[randomList[counter]] = Af_rgb
		train_set_y[randomList[counter]] = 1.0
		counter = counter + 1

	# Load Asian images
	for number in range(1, train_images + 1):
		As_rgb = iio.imread("Dataset/Train/Resized_Images/Asian/Asian_" + str(number) + ".jpg")
		train_set_x[randomList[counter]] = As_rgb
		train_set_y[randomList[counter]] = 0.0
		counter = counter + 1

	print("Images loaded!")

	return train_set_x, train_set_y

if __name__ == '__main__':

	#Crop the images
	#cropimages()

	#Load the images into lists
	train_set_x, train_set_y = loadimages()

	#Convert the lists to arrays
	train_set_x = np.array(train_set_x, dtype="object")
	train_set_y = np.array(train_set_y, dtype="object")

	#Print some usefull parameters
	print("Number of training examples...: " + str(len(train_set_y)))
	print("Image chapes..................: " + str(train_set_x[0].shape))
	print("Train_set_x shape.............: " + str(train_set_x.shape))
	print("Train_set_y_shape.............: " + str(train_set_y.shape))
