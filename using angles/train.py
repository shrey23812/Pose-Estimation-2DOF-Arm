from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dense, Dropout, Conv2D,  Cropping2D
from keras.models import load_model
from keras.optimizers import Adam
import cv2
import numpy as np
from os import listdir, path
import matplotlib.pyplot as plt

IMAGE_WIDTH = 500
IMAGE_HEIGHT = 250
IMAGE_CHANNELS = 3

# initialize parameters
DATA_DIR = "./images/"
TEST_DIR = "./inv_images/"

BATCH_SIZE = 32
NUM_EPOCHS = 20

images = []
measurements = []

# Load the images
for f in listdir(DATA_DIR):
	images.append(cv2.imread(DATA_DIR + f))
	var = f.strip("[].jpg").split(",")
	measurements.append([float(var[0]),float(var[1])])

# Define numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

# Print out the shapes of the training data 
print(X_train.shape)
print(y_train.shape)

#shuffle X_train and y_train for validation split
s = np.arange(X_train.shape[0])
np.random.shuffle(s)

X_train = X_train[s]
y_train = y_train[s]

# build the model
model = Sequential()

filename = "model.h5"

if (not path.exists(filename)):

	print ("Creating new model")
	# Normalize image
	model.add(Lambda( lambda x: x/127.5 - 1.0 , input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))

	model.add( Conv2D( 24, 5, 5, subsample=(2,2), activation = 'relu' ) )
	model.add( Conv2D( 36, 5, 5, subsample=(2,2), activation = 'relu' ) )
	model.add( Conv2D( 48, 5, 5, subsample=(2,2), activation = 'relu' ) )
	model.add( Conv2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
	model.add( Conv2D( 128, 3, 3, subsample=(1,1), activation = 'relu' ) )
	model.add( Flatten() )
	# model.add(Dropout(0.8))
	model.add( Dense( 100, activation = 'relu' ) )
	model.add( Dense( 50, activation = 'relu') )
	model.add( Dense( 10, activation = 'relu' ) )
	model.add( Dense( 2 ) )

	# model.add(Lambda( lambda x: x/127.5 - 1.0 , input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)))
	# model.add( Conv2D( 24, 5, 5, subsample=(2,2), activation = 'relu' ) )
	# model.add( Conv2D( 36, 5, 5, subsample=(2,2), activation = 'relu' ) )
	# model.add( Conv2D( 48, 5, 5, subsample=(2,2), activation = 'relu' ) )
	# model.add( Conv2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
	# model.add( Conv2D( 96, 3, 3, subsample=(1,1), activation = 'relu' ) )
	# model.add( Conv2D( 128, 3, 3, subsample=(1,1), activation = 'relu' ) )
	# model.add( Flatten() )
	# # model.add(Dropout(0.8))
	# model.add( Dense( 100, activation = 'relu' ) )
	# model.add( Dense( 50, activation = 'relu') )
	# model.add( Dense( 10, activation = 'relu' ) )
	# model.add( Dense( 1 ) )
	
	# Print Model Summary
	#model.summary()

	# Compile Model
	model.compile(optimizer=Adam(lr=1e-3), loss="mse")
	model.fit(X_train, y_train, shuffle= True, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,validation_split = 0.1)
	#save the trained model as h5 file
	print("Saving model")
	model.save(filename, overwrite=True)
	
else:
	print ("Using Existing Model")
	model = load_model(filename)

# Run predictions
print("Running prediction")
x=[]
y=[]
i = 0
for f in listdir(DATA_DIR):
	img = cv2.imread(DATA_DIR + f)
	img = img.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
	img = np.array(img)

	var = f.strip("[].jpg").split(",")
	print(var, " : ", model.predict(img)[0],i,abs(np.array(var,dtype="float32")-model.predict(img)[0]))#,abs(np.array(var,dtype="float32")-np.array(model.predict(img)[0])))
	x.append(np.sqrt(np.power(np.array(var,dtype="float32")-model.predict(img)[0],2)))
	y.append(i)
	i = i+1
error = np.sum(x,axis=0)/len(x)
print("error:",error)
plt.plot(x,y)
plt.show()

# for i in range(180):
# 	img = cv2.imread('./images/' + str(i) + '.jpg')
# 	img = img.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
# 	img = np.array(img)

# 	print(i, " : ", model.predict(img)[0])
# 	x.append(i)
# 	y.append(np.abs(i-model.predict(img)[0]))
# fig, ax = plt.subplots()
# ax.plot(x, y)
# # fig.savefig("360.png")
# plt.show()