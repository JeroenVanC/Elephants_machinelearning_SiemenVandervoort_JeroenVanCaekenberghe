import numpy as np
import random

from numpy.core.numeric import count_nonzero
import utils
from scipy import optimize
import imageio as iio
#import pdb


train_images = 50 # 400
counter_cost = 0

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
   # pdb.set_trace()
    # Load African images
    for number in range(1, train_images + 1):
        
        Af_rgb = iio.imread("Dataset\Train\Resized_Images_20_20\African\African_" + str(number) + ".jpg")
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

def sigmoid(z):
    return 1/(1+np.exp(-z))

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    global counter_cost
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = y.size

    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    a1 = np.concatenate([np.ones((m, 1)), X], axis=1)

    a2 = sigmoid(a1.dot(Theta1.T))
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    
    a3 = sigmoid(a2.dot(Theta2.T))

    
    y_matrix = y.reshape(-1)
    y_matrix = y_matrix.astype(int)
    y_matrix = np.eye(num_labels)[y_matrix]
    
    temp1 = Theta1
    temp2 = Theta2
    
    # Add regularization term
    
    reg_term = (lambda_ / (2 * m)) * (np.sum(np.square(temp1[:, 1:])) + np.sum(np.square(temp2[:, 1:])))
    
    J = (-1 / m) * np.sum((np.log(a3) * y_matrix) + np.log(1 - a3) * (1 - y_matrix)) + reg_term
    # Backpropogation
    
    delta_3 = a3 - y_matrix
    delta_2 = delta_3.dot(Theta2)[:, 1:] * sigmoidGradient(a1.dot(Theta1.T))

    Delta1 = delta_2.T.dot(a1)
    Delta2 = delta_3.T.dot(a2)
    
    # Add regularization to gradient

    Theta1_grad = (1 / m) * Delta1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_ / m) * Theta1[:, 1:]
    
    Theta2_grad = (1 / m) * Delta2
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_ / m) * Theta2[:, 1:]
    
    # ===================== Alterntate solutions =====================
    # my_final_matrix = np.zeros(a3.shape)
    # for c in np.arange(num_labels):
    #    my_final_matrix[:, c] = (np.log(a3[:, c]) * (y == c)) + (np.log(1 - a3[:, c]) * (1 - (y == c)))
    #J = (-1 / m) * np.sum(my_final_matrix)
    # ================================================================
    
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

#    print("|| It: " + str(counter_cost) + " || J: " + str(J) + " || grad: " +str(grad) + " ||")
    print("|| It: " + str(counter_cost) + " || J: " + str(J) + " ||")

    counter_cost = counter_cost + 1

    return J, grad

def sigmoidGradient(z):

    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================

    g = sigmoid(z) * (1 - sigmoid(z))

    # =============================================================
    return g

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):

    # You need to return the following variables correctly 
    W = np.zeros((L_out, 1 + L_in))

    # ====================== YOUR CODE HERE ======================
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init


    # ============================================================
    return W




def predict(Theta1, Theta2, X):
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    # useful variables
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])

    # ====================== YOUR CODE HERE ======================
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    
    a2 = sigmoid(X.dot(Theta1.T))
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
    
    p = np.argmax(sigmoid(a2.dot(Theta2.T)), axis = 1)

    # =============================================================
    return p

if __name__ == '__main__':

    # Setup parameters you will use for this NN ---------------------
    input_layer_size = 1200 #50x50 pixels for the input images * 3 for rgb value
    hidden_layer_size = 800 # 2/3 input layer
    num_labels = 2          # 1 label, african or asian
    #----------------------------------------------------------------
    X, y = loadimages()
    X = np.array(X, dtype="object")
    y = np.array(y, dtype="object")
    X = X.astype(float)
    y = y.astype(float)
    X = X.reshape(X.shape[0], -1)
    X = X /255


    # Initializing NEural Network Parameters ------------------------
    print('Initializing Neural Network Parameters ...')
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    print('theta 1 done')
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    print('theta 2 done')

    # Unroll parameters
    initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)
    print('Unroll Parameters')
    #---------------------------------------------------------------------



    #  After you have completed the assignment, change the maxiter to a larger
    #  value to see how more training helps.
    options= {'maxiter': 50}

    #  You should also try different values of lambda
    lambda_ = 1

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                            hidden_layer_size,
                                            num_labels, X, y, lambda_)
    print('costfunction done')
    
    # Now, costFunction is a function that takes in only one argument
    # (the neural network parameters)
    res = optimize.minimize(costFunction,
                            initial_nn_params,
                            jac=True,
                            method='TNC',
                            options=options)

    # get the solution of the optimization
    nn_params = res.x
            
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    Theta1 = Theta1.astype(float)

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1))) 
    Theta2 = Theta2.astype(float)

    #Run nieuwe foto's:
    x_test = []
    for number in range(100, 201):
        image_rgb = []
        As_rgb = iio.imread('C:/Users/HP/Documents/SCHOOL/Master_Elektronica_ICT/Machine_Learning/Project_Github/Elephants_machinelearning_SiemenVandervoort_JeroenVanCaekenberghe/Code/Dataset/Train/Resized_Images_20_20/African/African_' + str(number) + '.jpg')
        x_test.append(As_rgb)

    x_test = np.array(x_test, dtype="object")
    x_test = x_test.astype(float)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_test = x_test/255

    prediction = utils.predict(Theta1, Theta2, x_test)
    print(str(prediction))
    print('Training Set Accuracy: %f' % (np.mean(prediction == y) * 100))

    african = 0
    asian = 0
    for result in prediction:
        if result == 1:
            african = african + 1
        else:
            asian = asian + 1

    print("Africans: " + str(african) + ", Asians: " + str(asian))
    print("% Afr   : " + str(round(african/(african+asian), 2)) + ", % Asi: " + str(round(asian/(asian+african), 2)))




