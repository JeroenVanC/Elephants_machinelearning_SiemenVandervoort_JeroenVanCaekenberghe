#This file contains the helper functions

#import
import Logistic_regression as logreg
import Neural_network as neunet
import Neural_network_library as neunetlib

#Global parameters
newSize = 50

#Return input text
def getInputText(number):
	if number == "1":
		return "You choose Logistic Regression."
	elif number == "2":
		return "You choose the Stand Alone Neural Network."
	elif number == "3":
		return "You choose the Neural Network with libraries."
	elif number == "4":
		return "You choose to crop images, I can't do it by myself..."
	elif number == "5":
		return "You choose to mirror images, I can't do it by myself..."
	else:
		return "Error with returning input text"

#Start choosen algorithm
def startAlgorithm(algorithm, train_images, iterations):
	if algorithm == "1":
		logreg.main(int(train_images), int(iterations))
	elif algorithm == "2":
		neunet.main(int(train_images))
	elif algorithm == "3":
		neunetlib.main(int(train_images))
	else:
		print("Error with starting machine learning algorithm!")

#Crop images
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

#Mirror images
def mirrorImages():
	print("Mirroring images ...")

	#Rotate African images to get more training data
	for number in range(1, train_images):
		image = Image.open("Dataset/Train/Resized_Images/African/African_" + str(number) + ".jpg")
		rotated_image = image.transpose(Image.FLIP_LEFT_RIGHT)
		rotated_image.save("Dataset/Train/Resized_Images/African/African_" + str(number + train_images) + ".jpg")

	#Rotate Asian images to get more training data
	for number in range(1, train_images):
		image = Image.open("Dataset/Train/Resized_Images/Asian/Asian_" + str(number) + ".jpg")
		rotated_image = image.transpose(Image.FLIP_LEFT_RIGHT)
		rotated_image.save("Dataset/Train/Resized_Images/Asian/Asian_" + str(number + train_images) + ".jpg")

	print("Images mirrored!")