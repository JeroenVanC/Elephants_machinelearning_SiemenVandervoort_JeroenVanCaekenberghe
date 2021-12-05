#imports
import Utils_nieuw as utils
import os

if __name__ == '__main__':
	#Clear the command window
	os.system('cls')

	#Print the starting interface
	print("***************************************************")
	print("*            Made by Jeroen and Siemen            *")
	print("*           Machine Learning Course 2021          *")
	print("* Welcome to the Elephant Classification program! *")
	print("***************************************************")

	print("                                                   ")

	print("A: Choose one of the options below: ")
	print("	1: Logistic Regression Model")
	print("	2: Stand Alone Neural Network")
	print("	3: Neural Network with libraries")
	print("	4: Crop images")
	print("	5: Mirror images")

	option = input()
	print("	" + utils.getInputText(option))

	print("                                                   ")

	if option >= "1" and option <= "3":
		print("B: Choose the amount of training images: ")
		train_images = input()
		print("C: Choose the amount of iterations: ")
		iterations = input()

		print("                                                   ")

		#Start machine learning algorithms
		utils.startAlgorithm(option, train_images, iterations)