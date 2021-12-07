# Import
from PIL import Image
import imageio as iio
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Constants
newSize = 50
#train_images = 400
input_layer_size = newSize * newSize * 3
hidden_layer_size = 25
output_layer_size = 1
elephant_labels = 2

def loadimages(train_images):
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

def loadTestset(start, end):
	print("Loading test images ...")
	africans = []
	asians = []

	# Load African test images
	for number in range(start, end):
		Af_rgb = iio.imread("Dataset/Train/Resized_Images_20_20/African/African_" + str(number) + ".jpg")
		africans.append(Af_rgb)

	# Load Asian test images
	for number in range(start, end):
		As_rgb = iio.imread("Dataset/Train/Resized_Images_20_20/Asian/Asian_" + str(number) + ".jpg")
		asians.append(As_rgb)

	africans = np.array(africans, dtype="object")
	asians = np.array(asians, dtype="object")

	africans = africans.astype(float)
	asians = asians.astype(float)

	africans = africans/255
	asians = asians/255

	africans = africans.reshape(africans.shape[0], -1)
	asians = asians.reshape(asians.shape[0], -1)

	print("Test images loaded!")

	return africans, asians


def main(train_images, iterations):
	#Load the images into lists
	train_set_x, train_set_y = loadimages(train_images)

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

	print("Train the model")
	model = LogisticRegression(C=0.2, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=iterations,
                   multi_class='ovr', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)

	model.fit(train_set_x,train_set_y)

	#Load test images
	africans, asians = loadTestset(401, 511)

	africans_pred = model.predict(africans)
	asians_pred = model.predict(asians)

	#Test African images
	africans = 0
	asians = 0
	for number in range(0, len(africans_pred)):
		if africans_pred[number] == 1:
			africans = africans + 1
		else:
			asians = asians + 1

	print("Africans: " + str(africans) + ", Asians: " + str(asians))
	print("% Afr   : " + str(round(africans/(africans+asians), 2)) + ", % Asi: " + str(round(asians/(asians+africans), 2)))

	#Test Asian images
	africans = 0
	asians = 0
	for number in range(0, len(asians_pred)):
		if asians_pred[number] == 1:
			africans = africans + 1
		else:
			asians = asians + 1

	print("Africans: " + str(africans) + ", Asians: " + str(asians))
	print("% Afr   : " + str(round(africans/(africans+asians), 2)) + ", % Asi: " + str(round(asians/(asians+africans), 2)))













