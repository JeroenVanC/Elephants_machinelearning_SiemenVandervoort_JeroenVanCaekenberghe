# Import
from PIL import Image
import imageio as iio
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score

# Constants
newSize = 50
#train_images = 400
input_layer_size = newSize * newSize * 3
hidden_layer_size = 25
output_layer_size = 1
elephant_labels = 2

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

	print("Images loaded!")

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

def loadTestset(start, end):
	print("Loading test images ...")
	test_set_x = []
	test_set_y = []
	asians = []
	africans = []

	# Load African test images
	for number in range(start, end):
		Af_rgb = iio.imread("Dataset/Train/Resized_Images_20_20/African/African_" + str(number) + ".jpg")
		test_set_x.append(Af_rgb)
		test_set_y.append(1.0)

	# Load Asian test images
	for number in range(start, end):
		As_rgb = iio.imread("Dataset/Train/Resized_Images_20_20/Asian/Asian_" + str(number) + ".jpg")
		test_set_x.append(As_rgb)
		test_set_y.append(0.0)

	test_set_x = np.array(test_set_x, dtype="object")
	test_set_y = np.array(test_set_y, dtype="object")

	test_set_x = test_set_x.astype(float)
	test_set_y = test_set_y.astype(float)

	test_set_x = test_set_x/255

	test_set_x = test_set_x.reshape(test_set_x.shape[0], -1)

	print("Test images loaded!")

	return test_set_x, test_set_y


#def main(train_images, iterations):
if __name__ == '__main__':
	train_images = 400
	iterations = 1000

	#Load the images into lists
	train_set_x, train_set_y = loadimages_alternating(train_images)

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

	train_set_x = train_set_x/255

	train_set_x = train_set_x.reshape(train_set_x.shape[0], -1)

	#Parameters to change
	param_C     = [0.01, 0.03, 0.06, 0.09, 0.1, 0.3, 0.6, 0.9, 1]
	param_iterations = [10000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]
	param_C_counter = 0
	param_iterations_counter = 0

	#Load test images
	test_set_x, test_set_y = loadTestset(401, 511)

	for tests in range(0,1):
		print("Train the model")
		model = LogisticRegression(C=param_C[param_C_counter], class_weight=None, dual=False, fit_intercept=True,
					   intercept_scaling=1, l1_ratio=None, max_iter=param_iterations[param_iterations_counter],
					   multi_class='ovr', n_jobs=None, penalty='l2',
					   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
					   warm_start=False)

		model.fit(train_set_x, train_set_y)

		#Accuracy train set
		accuracy_train_set = model.score(train_set_x, train_set_y)
		print("Train Set Accuracy    : " + str(accuracy_train_set))

		#Accuracy test set
		accuracy_test_set = model.score(test_set_x, test_set_y)
		print("Test Set Accuracy     : " + str(accuracy_test_set))

		print("Iterations            : " + str(param_iterations[param_iterations_counter]))
		print("C                     : " + str(param_C[param_C_counter]))

		#Export results
		path = "Output_parameters/LogRegLib_iterations_" + str(param_iterations[param_iterations_counter]) + "_C_" + str(param_C[param_C_counter])

		np.savez(str(path), train_images = train_images, 
					param_iterations = param_iterations[param_iterations_counter], 
					accuracy_train_set = accuracy_train_set,
					accuracy_test_set = accuracy_test_set,
					param_C = param_C[param_C_counter]
				)

		#---------------------------------------------#
		#Load exported data
		"""
		a = np.load(str(path) + ".npz", allow_pickle=True)
		print(str(a["param_lambda"]))
		print(str(a["w"]))
		"""











