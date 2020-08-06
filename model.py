import h5py
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, Flatten, Conv2D,MaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix as cf

def load_dataset():
    """
    Reads the h5py file containing the dataset and returns the training and test set
    """
    hf=h5py.File("image_final.h5",'r')

    X_train_orig=np.array(hf.get("X_train_orig"))
    X_test_orig=np.array(hf.get("X_test_orig"))
    Y_train_orig=np.array(hf.get("Y_train_orig"))
    Y_test_orig=np.array(hf.get("Y_test_orig"))
    # Reshape the dataset 
    Y_train_orig=Y_train_orig.reshape(Y_train_orig.shape[0],1)
    Y_test_orig=Y_test_orig.reshape(Y_test_orig.shape[0],1)

    hf.close()

    return X_train_orig,X_test_orig,Y_train_orig,Y_test_orig

def model_nn(input_shape=(64,64,1)):
    
    # Input Layer
	X_input=Input(input_shape)
	# 32 Filter convolution layer each of size 3x3
	X=Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,1))(X_input)
	# Max Pooling layer
	X=MaxPooling2D(pool_size=(1,1))(X)
	# 32 Filter convolution layer each of size 3x3
	X=Conv2D(32,(3,3),activation='relu')(X)
	# Max Pooling layer
	X=MaxPooling2D(pool_size=(1,1))(X)
	#64 Filter convolution layer each of size 3x3
	X=Conv2D(64, (3, 3), activation='relu')(X)
	# Max Pooling layer
	X=MaxPooling2D(pool_size=(1,1))(X)
	# Dropout layer for convergence i.e randomly turn neurons on and off
	X=Dropout(0.25)(X)
	#flatten since too many dimensions, we only want a classification output
	X=Flatten()(X)
	#fully connected to get all relevant data
	X=Dense(128, activation='relu')(X)
	#Dropout layer for Convergence
	X=Dropout(0.5)(X)
	#output a sigmoid to squash the matrix into output probabilities
	X=Dense(1, activation='sigmoid')(X)
	model = Model(inputs = X_input, outputs = X,name="CNN")
    
	return model

model =model_nn(input_shape = (64, 64, 1))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Calling load_dataset to store the dataset
X_train_orig, X_test_orig, Y_train, Y_test = load_dataset()
# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

model.fit(X_train, Y_train, epochs = 15, batch_size = 32)

y_pred=model.predict(X_test)
# The output of the model is array of real numbers therefore values greater than 0.5 will be evaluated as 1 otherwise 0
y_pred=(y_pred>0.5)
# Confusion Matrix
cf(Y_test,y_pred)
# Save the model for further use
model.save('models/CNN_Model.h5', overwrite=True)


