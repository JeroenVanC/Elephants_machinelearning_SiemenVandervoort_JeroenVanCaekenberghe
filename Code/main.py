# Import
from PIL import Image
import imageio as iio
import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
newSize = 50
train_images = 900
input_layer_size = newSize * newSize * 3
hidden_layer_size = 25
output_layer_size = 1
elephant_labels = 2

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

def sigmoid(z):
	return 1/(1+np.exp(-z))

def propagation(w, b, x, y):
	m = x.shape[1]

	#Forward propagation to find the cost
	A = sigmoid(np.dot(w.T, x) + b)       # compute activation
	cost = -1/m * np.sum(y*np.log(A) + (1 - y)*np.log(1 - A))  # compute cost

	#Backward propagation to find gradient
	dw = 1/m * np.dot(x, (A - y).T)
	db = 1/m * np.sum(A - y)

	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	return dw, db, cost

def optimizeGradientDescent(w, b, x, y, iterations, alpha):
	costs = []

	for i in range(iterations):
		#Propagation
		dw, db, cost = propagation(w, b, x, y)

		#Update w and b
		w = w - alpha * dw
		b = b - alpha * db

		# Record the costs
		if i % 100 == 0:
			costs.append(cost)

		if True and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))

	return w, b, dw, db, costs

def model(train_set_x, train_set_y, iterations, alpha):
	#Initialize values
	dimension = train_set_x.shape[0]
	w = np.zeros((dimension, 1))
	b = 0

	assert(w.shape == (dimension, 1))
	assert(isinstance(b, float) or isinstance(b, int))

	#Gradient descent
	w, b, dw, db, costs = optimizeGradientDescent(w, b, train_set_x, train_set_y, iterations, alpha)

	#Print accuracy
	Y_prediction_train = predict(w, b, train_set_x)
	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))

	info = {"costs": costs,
		 "Y_prediction_train" : Y_prediction_train, 
		 "w" : w, 
		 "b" : b,
		 "alpha" : alpha,
		 "iterations": iterations}

	return info

def predict(w, b, x):
	m = x.shape[1]
	Y_prediction = np.zeros((1,m))
	w = w.reshape(x.shape[0], 1)

	# Compute vector "A" predicting the probabilities of an African elephant being present in the picture
	A = sigmoid(np.dot(w.T, x) + b)

	for i in range(A.shape[1]):
		# Convert probabilities A[0,i] to actual predictions p[0,i]
		if A[:, i] <= 0.5:
			Y_prediction[:, i] = 0
		else:
			Y_prediction[:, i] = 1

	assert(Y_prediction.shape == (1, m))

	return Y_prediction

def own_Image(my_image, info):
	# We preprocess the image to fit your algorithm.
	image_rgb = []

	As_rgb = iio.imread(my_image)
	image_rgb.append(As_rgb)
	image_rgb = np.array(image_rgb, dtype="object")

	image_rgb_flatten = image_rgb.reshape(image_rgb.shape[0], -1).T

	image_rgb_flatten = image_rgb_flatten/255

	image_rgb_flatten = image_rgb_flatten.astype(float)

	my_predicted_image = predict(info["w"], info["b"], image_rgb_flatten)

	#plt.imshow(image_rgb)
	#print("y = " + str(np.squeeze(my_predicted_image)))

	#file.write("Result: " + str(np.squeeze(my_predicted_image)) + "\n")µ

	return np.squeeze(my_predicted_image)


if __name__ == '__main__':

	#Preproces the images
	#cropimages()
	#mirrorImages()

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

	#Reshape the arrays
	train_set_y = train_set_y.astype(float)
	train_set_x = train_set_x.astype(float)

	train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1).T

	#Standardize the dataset
	train_set_x = train_set_x_flatten/255

	#Train the model
	info = model(train_set_x, train_set_y, 10000, 0.005)

	#Plot costs
	costs = np.squeeze(info['costs'])
	plt.plot(costs)
	plt.ylabel('Cost')
	plt.xlabel('Iterations (per hundreds)')
	plt.title("Learning rate =" + str(0.005))
	plt.show()

	#Test an image
	africans = 0
	asians = 0
	for number in range(901, 1001):
		result = own_Image('C:/Users/HP/Documents/SCHOOL/Master_Elektronica_ICT/Machine_Learning/Project_Github/Elephants_machinelearning_SiemenVandervoort_JeroenVanCaekenberghe/Code/Dataset/Train/Resized_Images/African/African_' + str(number) + '.jpg', info)
		if result == 1:
			africans = africans + 1
		else:
			asians = asians + 1

	print("Africans: " + str(africans) + ", Asians: " + str(asians))
	print("% Afr   : " + str(round(africans/(africans+asians), 2)) + ", % Asi: " + str(round(asians/(asians+africans), 2)))

