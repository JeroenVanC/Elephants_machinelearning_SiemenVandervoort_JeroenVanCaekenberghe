# Import
from PIL import Image
import imageio as iio
import numpy as np
import random
import matplotlib.pyplot as plt
import json

#Load the images randomly
def loadimages_randomly(train_images):
	print("Loading " + str(train_images) + " images...")
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
		Af_rgb = iio.imread("Dataset/Train/Resized_Images_20_20/African/African_" + str(number) + ".jpg")
		train_set_x[randomList[counter]] = Af_rgb
		train_set_y[randomList[counter]] = 1.0
		counter = counter + 1

	# Load Asian images
	for number in range(1, train_images + 1):
		As_rgb = iio.imread("Dataset/Train/Resized_Images_20_20/Asian/Asian_" + str(number) + ".jpg")
		train_set_x[randomList[counter]] = As_rgb
		train_set_y[randomList[counter]] = 0.0
		counter = counter + 1

	print("	Images loaded!")

	return train_set_x, train_set_y

#Load images afwisselend
def loadimages_alternating(train_images):
	print("Loading " + str(train_images) + " images...")
	train_set_x = []
	train_set_y = []

	for i in range(0, 2 * train_images):
		train_set_x.append(0)
		train_set_y.append(0)

	#Use a counter to iterate through the list
	counter = 0

	# Load African images
	for number in range(1, train_images + 1):
		Af_rgb = iio.imread("Dataset/Train/Resized_Images_20_20/African/African_" + str(number) + ".jpg")
		train_set_x[counter] = Af_rgb
		train_set_y[counter] = 1.0
		counter = counter + 2

	counter = 1
	# Load Asian images
	for number in range(1, train_images + 1):
		As_rgb = iio.imread("Dataset/Train/Resized_Images_20_20/Asian/Asian_" + str(number) + ".jpg")
		train_set_x[counter] = As_rgb
		train_set_y[counter] = 0.0
		counter = counter + 2

	print("	Images loaded!")

	return train_set_x, train_set_y

#Calculate the sigmoid fuction
def sigmoid(z):
	return 1/(1+np.exp(-z))

#Do the propagation to calculate the cost
def propagation(w, b, x, y):
	m = x.shape[1]

	#Forward propagation to find the cost
	A = sigmoid(np.dot(w.T, x) + b)

	cost = -1/m * np.sum(y*np.log(A) + (1 - y)*np.log(1 - A))

	#Backward propagation to find gradient
	dw = 1/m * np.dot(x, (A - y).T)
	db = 1/m * np.sum(A - y)

	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	return dw, db, cost

#Calculate the thetas
def optimizeGradientDescent(w, b, x, y, iterations, alpha):
	costs = []
	costs_test_set = []
	accuracy_test_set = []
	accuracy_train_set = []

	test_set_x, test_set_y = getOwnImages()

	for i in range(iterations):
		#Propagation
		dw, db, cost = propagation(w, b, x, y)

		dw2, db2, test_set_cost = propagation(w, b, test_set_x, test_set_y)

		#Update w and b
		w = w - alpha * dw
		b = b - alpha * db

		# Record the costs
		if i % 100 == 0:
			costs.append(cost)
			costs_test_set.append(test_set_cost)

		if True and i % 100 == 0:
			accuracy_test = getAccuracyTestSet(w, b)
			accuracy_train, Y_prediction_train = getAccuracyTrainSet(w, b, x, y)
			accuracy_test_set.append(accuracy_test)
			accuracy_train_set.append(accuracy_train)

			print("Train set cost after iteration %i: %f" %(i, cost) + ",	Test set accuracy: " + str(np.round(accuracy_test, 5)) + ",	Train set accuracy: " + str(np.round(accuracy_train, 5)) + ",	Test set cost: " + str(test_set_cost) + ",	Learning rate: " + str(alpha))

	return w, b, dw, db, costs, accuracy_test_set, accuracy_train_set, costs_test_set

#Overall model
def model(train_set_x, train_set_y, iterations, alpha):
	#Initialize values
	dimension = train_set_x.shape[0]
	w = np.zeros((dimension, 1))
	b = 0

	assert(w.shape == (dimension, 1))
	assert(isinstance(b, float) or isinstance(b, int))

	#Gradient descent
	w, b, dw, db, costs, accuracy_test_set, accuracy_train_set, costs_test_set = optimizeGradientDescent(w, b, train_set_x, train_set_y, iterations, alpha)

	#Print accuracy
	accuracy_train, Y_prediction_train = getAccuracyTrainSet(w, b, train_set_x, train_set_y)
	accuracy_test = getAccuracyTestSet(w, b)

	print('Training Set Accuracy: %f' % accuracy_train)
	print('Test Set Accuracy    : %f' % accuracy_test)

	info = {"costs": costs,
		 "costs_test_set": costs_test_set,
		 "Y_prediction_train" : Y_prediction_train, 
		 "w" : w, 
		 "b" : b,
		 "alpha" : alpha,
		 "iterations": iterations,
		 "accuracy_train": accuracy_train,
		 "accuracy_test": accuracy_test,
		 "accuracy_test_set": accuracy_test_set,
		 "accuracy_train_set": accuracy_train_set
		 }

	return info

#Do the prediction
def predict(w, b, x):
	m = x.shape[1]
	Y_prediction = np.zeros((1,m))
	w = w.reshape(x.shape[0], 1)

	# Compute vector "A" predicting the probabilities of an African elephant being present in the picture
	A = sigmoid(np.dot(w.T, x) + b)
	A = sigmoid(np.dot(w.T, x))

	for i in range(A.shape[1]):
		# Convert probabilities A[0,i] to actual predictions p[0,i]
		if A[:, i] <= 0.5:
			Y_prediction[:, i] = 0
		else:
			Y_prediction[:, i] = 1

	assert(Y_prediction.shape == (1, m))

	return Y_prediction

#Try our own images
def own_Image(my_image, w ,b):
	#We preprocess the image to fit the algorithm.
	image_rgb = []

	As_rgb = iio.imread(my_image)
	image_rgb.append(As_rgb)
	image_rgb = np.array(image_rgb, dtype="object")

	image_rgb_flatten = image_rgb.reshape(image_rgb.shape[0], -1).T

	image_rgb_flatten = image_rgb_flatten/255

	image_rgb_flatten = image_rgb_flatten.astype(float)

	my_predicted_image = predict(w, b, image_rgb_flatten)

	return np.squeeze(my_predicted_image)

def getOwnImages():
	test_images = 110
	test_set_x = []
	test_set_y = []

	for i in range(0, 2 * test_images):
		test_set_x.append(0)
		test_set_y.append(0)

	#Use a counter to iterate through the list
	counter = 0

	# Load African images
	for number in range(401, 511):
		Af_rgb = iio.imread("C:/Users/HP/Documents/SCHOOL/Master_Elektronica_ICT/Machine_Learning/Project_Github/Elephants_machinelearning_SiemenVandervoort_JeroenVanCaekenberghe/Code/Dataset/Train/Resized_Images_20_20/African/African_" + str(number) + ".jpg")
		test_set_x[counter] = Af_rgb
		test_set_y[counter] = 1.0
		counter = counter + 2

	counter = 1
	# Load Asian images
	for number in range(401, 511):
		As_rgb = iio.imread('C:/Users/HP/Documents/SCHOOL/Master_Elektronica_ICT/Machine_Learning/Project_Github/Elephants_machinelearning_SiemenVandervoort_JeroenVanCaekenberghe/Code/Dataset/Train/Resized_Images_20_20/Asian/Asian_' + str(number) + '.jpg')
		test_set_x[counter] = As_rgb
		test_set_y[counter] = 0.0
		counter = counter + 2

	#Convert the lists to arrays
	test_set_x = np.array(test_set_x, dtype="object")
	test_set_y = np.array(test_set_y, dtype="object")

	#Reshape the arrays
	test_set_y = test_set_y.astype(float)
	test_set_x = test_set_x.astype(float)

	train_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T

	#Standardize the dataset
	test_set_x = train_set_x_flatten/255

	return test_set_x, test_set_y


def getAccuracyTrainSet(w, b, train_set_x, train_set_y):
	Y_prediction_train = predict(w, b, train_set_x)

	accuracy = np.mean(Y_prediction_train == train_set_y) * 100

	return accuracy, Y_prediction_train

def getAccuracyTestSet(w, b):
	#Test African images
	test_set_y = []
	y_prediction_test = []
	for number in range(401, 511):
		y_prediction_test.append(str(own_Image('C:/Users/HP/Documents/SCHOOL/Master_Elektronica_ICT/Machine_Learning/Project_Github/Elephants_machinelearning_SiemenVandervoort_JeroenVanCaekenberghe/Code/Dataset/Train/Resized_Images_20_20/African/African_' + str(number) + '.jpg', w, b)))
		test_set_y.append(1.0)

	#Test Asian images
	for number in range(401, 511):
		y_prediction_test.append(str(own_Image('C:/Users/HP/Documents/SCHOOL/Master_Elektronica_ICT/Machine_Learning/Project_Github/Elephants_machinelearning_SiemenVandervoort_JeroenVanCaekenberghe/Code/Dataset/Train/Resized_Images_20_20/Asian/Asian_' + str(number) + '.jpg', w, b)))
		test_set_y.append(0.0)

	#Calculate accuracy
	correct = 0
	for counter in range(0, 220):
		if str(y_prediction_test[counter]) == str(test_set_y[counter]):
			correct = correct + 1

	accuracy_test_set = (correct/220) * 100

	return accuracy_test_set

def reloadExportedData():
	#---------------------------------------------#
	#Load exported data
	param_iterations = 50000
	param_lambda     = 0.005

	test_cost_to_plot = []
	train_cost_to_plot = []
	test_acc_to_plot = []
	train_acc_to_plot = []

	#Plot cost in function of lambda
	param_lambda = [0.0006, 0.0009, 0.001, 0.003, 0.006, 0.009, 0.01, 0.03]
	plt.figure()
	for learningRate in param_lambda:
		path = "Output_parameters/LogReg_iterations_" + str(param_iterations) + "_lamda_" + str(learningRate)
		exported_data = np.load(str(path) + ".npz", allow_pickle=True)

		train_images = exported_data["train_images"]
		param_iterations = exported_data["param_iterations"]
		accuracy_train = exported_data["accuracy_train"]
		accuracy_test = exported_data["accuracy_test"]
		accuracy_test_set = exported_data["accuracy_test_set"]
		accuracy_train_set = exported_data["accuracy_train_set"]
		costs_train_set = exported_data["cost"]
		costs_test_set = exported_data["costs_test_set"]
		w = exported_data["w"]
		b = exported_data["b"]
		param_lambda_reload = exported_data["param_lambda"]

		print("Train set accuracy: " + str(accuracy_train) + ", 	Test set accuracy: " + str(accuracy_test))

		train_cost_to_plot.append(costs_train_set[len(costs_train_set)-1])
		test_cost_to_plot.append(costs_test_set[len(costs_test_set)-1])
		test_acc_to_plot.append(accuracy_test)
		train_acc_to_plot.append(accuracy_train)

		#plt.plot(costs_train_set, label = "Train set cost " + str(param_lambda_reload))
		plt.plot(costs_test_set, label = "Test set cost " + str(param_lambda_reload))
		plt.ylabel('Cost')
		plt.xlabel('Iterations (hundreds)')
		plt.legend()

	#Cost in functie van lamda
	plt.figure()
	plt.plot(param_lambda, train_cost_to_plot, label = "Train set cost")
	plt.plot(param_lambda, test_cost_to_plot, label = "Test set cost")
	plt.ylabel('Cost')
	plt.xlabel('Learning rate')
	#plt.xscale('log')
	plt.legend()
	#plt.title("Learning rate = " + str(param_lambda))
	plt.title("Changing learning rate")

	#Accuracy in functie van lambda
	plt.figure()
	plt.plot(param_lambda, train_acc_to_plot, label = "Accuracy train set")
	plt.plot(param_lambda, test_acc_to_plot, label = "Accuracy test set")
	plt.ylabel('Accuracy (%)')
	plt.xlabel('Learning rate')
	plt.xscale('log')
	plt.legend()
	#plt.title("Learning rate = " + str(param_lambda))
	plt.title("Changing learning rate")

	plt.show()

#def main(train_images, iterations):
if __name__ == '__main__':
	
	reloadExportedData()
	"""
	train_images = 400

	#Load the images into lists
	train_set_x, train_set_y = loadimages_alternating(train_images)

	#Convert the lists to arrays
	train_set_x = np.array(train_set_x, dtype="object")
	train_set_y = np.array(train_set_y, dtype="object")

	#Print some usefull parameters
	print(" ")
	print("Number of training examples...: " + str(len(train_set_y)))
	print("Image chapes..................: " + str(train_set_x[0].shape))
	print("Train_set_x shape.............: " + str(train_set_x.shape))
	print("Train_set_y_shape.............: " + str(train_set_y.shape))
	print(" ")

	#Reshape the arrays
	train_set_y = train_set_y.astype(float)
	train_set_x = train_set_x.astype(float)

	train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1).T

	#Standardize the dataset
	train_set_x = train_set_x_flatten/255

	#Parameters to change
	param_lambda = [0.00001, 0.00003, 0.00006, 0.00009, 0.0001, 0.0003, 0.0006, 0.0009, 0.001]
	param_iterations = [500000]
	param_lambda_counter = 0
	param_iterations_counter = 0

	for tests in range(0, len(param_lambda)):
		param_lambda_counter = tests
		#Train the model
		info = model(train_set_x, train_set_y, param_iterations[param_iterations_counter], param_lambda[param_lambda_counter])

		#Get info
		costs = np.squeeze(info['costs'])
		costs_test_set = np.squeeze(info['costs_test_set'])
		accuracy_train = np.squeeze(info['accuracy_train'])
		accuracy_test = np.squeeze(info['accuracy_test'])
		w = np.squeeze(info['w'])
		b = np.squeeze(info['b'])
		accuracy_test_set = info['accuracy_test_set']
		accuracy_train_set = info['accuracy_train_set']

		#Plot and save costs & accuracy test set
		plt.figure()
		plt.plot(costs, label = "Train set cost")
		plt.plot(costs_test_set, label = "Test set cost")
		plt.ylabel('Cost')
		plt.xlabel('Iterations (per hundreds)')
		plt.legend()
		plt.title("Learning rate = " + str(param_lambda[param_lambda_counter]))
		plt.savefig("Output_parameters_Ultimate_Test/LogReg_iterations_" + str(param_iterations[param_iterations_counter]) + "_lamda_" + str(param_lambda[param_lambda_counter]) + "_costs.jpg")

		plt.figure()
		plt.plot(accuracy_test_set, label = "Accuracy test set")
		plt.plot(accuracy_train_set, label = "Accuracy train set")
		plt.ylabel('Accuracy (%)')
		plt.xlabel('Iterations (per hundreds)')
		plt.legend()
		plt.title("Learning rate = " + str(param_lambda[param_lambda_counter]))
		plt.savefig("Output_parameters_Ultimate_Test/LogReg_iterations_" + str(param_iterations[param_iterations_counter]) + "_lamda_" + str(param_lambda[param_lambda_counter]) + "_accuracy.jpg")

		#Export results
		path = "Output_parameters_Ultimate_Test/LogReg_iterations_" + str(param_iterations[param_iterations_counter]) + "_lamda_" + str(param_lambda[param_lambda_counter])

		np.savez(str(path), train_images = train_images, 
					param_iterations = param_iterations[param_iterations_counter], 
					accuracy_train = accuracy_train,
					accuracy_test = accuracy_test,
					accuracy_test_set = accuracy_test_set,
					accuracy_train_set = accuracy_train_set,
					cost = costs,
					costs_test_set = costs_test_set,
					w = w,
					b = b,
					param_lambda = param_lambda[param_lambda_counter]
				)
	"""








