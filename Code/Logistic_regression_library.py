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
import seaborn as sns

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
	param_C          = [0.00001, 0.00003, 0.00006, 0.00009, 0.0001, 0.0003, 0.0006, 0.0009, 0.001, 0.003, 0.006, 0.009, 0.01, 0.03, 0.06, 0.09, 0.1, 0.3, 0.6, 0.9, 1, 3, 6, 9, 10, 30, 60, 90, 100, 300, 600, 900, 1000, 3000, 6000, 9000, 10000]
	param_C_counter = 0.3
	param_iterations_counter = 100

	#Load test images
	test_set_x, test_set_y = loadTestset(401, 511)

	accuracy_test_set = []
	accuracy_train_set = []

	#for param_C_counter in param_C:
	print("Train the model")
	model = LogisticRegression(C=param_C_counter, class_weight=None, dual=False, fit_intercept=True,
				   intercept_scaling=1, l1_ratio=None, max_iter=param_iterations_counter,
				   multi_class='ovr', n_jobs=None, penalty='l2',
				   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
				   warm_start=False) #swarm_start=True means using previous settings to start again

	model.fit(train_set_x, train_set_y)

	#Accuracy train set
	accuracy_train = model.score(train_set_x, train_set_y) * 100
	print("Train Set Accuracy    : " + str(accuracy_train))
	accuracy_train_set.append(accuracy_train)

	#Accuracy test set
	accuracy_test = model.score(test_set_x, test_set_y) * 100
	print("Test Set Accuracy     : " + str(accuracy_test))
	accuracy_test_set.append(accuracy_test)

	print("Iterations            : " + str(param_iterations_counter))
	print("C                     : " + str(param_C_counter))
	print("-------------------------------------")

	
	result = model.predict(test_set_x)
	print(classification_report(test_set_y, result))

	cm=confusion_matrix(test_set_y, result)
	plt.figure(figsize=(12,6))
	plt.title("Confusion Matrix")
	sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
	plt.ylabel("Actual Values")
	plt.xlabel("Predicted Values")
	plt.show()
	"""

	#Accuracy
	plt.figure()
	plt.plot(param_C, accuracy_test_set, label = "Accuracy test set")
	plt.plot(param_C, accuracy_train_set, label = "Accuracy train set")
	plt.ylabel('Accuracy (%)')
	plt.xlabel('Inverse of regularization strength')
	plt.xscale('log')
	plt.legend()
	plt.title("Inverse of regularization strength = " + str(param_C_counter))
	plt.savefig("Output_parameters_lib/LogRegLib_iterations_" + str(param_iterations_counter) + "_C_" + str(param_C_counter) + "_accuracy.jpg")
	plt.show()
	
	#Export results
	path = "Output_parameters_lib/LogRegLib_iterations_" + str(param_iterations_counter) + "_C_" + str(param_C_counter)

	np.savez(str(path), train_images = train_images, 
				param_iterations = param_iterations_counter, 
				accuracy_train = accuracy_train,
				accuracy_test = accuracy_test,
				param_C = param_C,
				accuracy_test_set =accuracy_test_set,
				accuracy_train_set = accuracy_train_set
			)

	#---------------------------------------------#
	#Load exported data
	
	a = np.load(str(path) + ".npz", allow_pickle=True)
	print(str(a["param_lambda"]))
	print(str(a["w"]))
	"""











