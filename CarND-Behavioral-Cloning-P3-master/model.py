#Import the required libraries and functions
from os import path
from PIL import Image as PImage
from matplotlib import pyplot as plt
import numpy as np
from pandas import read_csv
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Lambda,Flatten
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error

#Read the csv file    
filename = 'driving_log.csv'
data = read_csv(filename,header=0)
array = data.values
y = array[:,3].astype('float32')

#create empty list for appending the names of images with zero steering angle
#create a matching list of steering angles
Images = []
y = []

for i in range(len(array)):
     if array[i][3] == 0:
        Images.append(array[i][7])
        y.append(array[i][3])
        
#Set seed for reproduceability. Randomly sample 12% from the list with zero
#steering angle. As the images with zero steering angle is overrepresented,
#this is done to undersample the images with zero steering angle.
random.seed = 7
Images = random.sample(Images,1300)
y = random.sample(y,1300)

#correcting parameter for left and right cameras
correction = 0.15

#Append the image list with left and right cameras
for i in range(len(array)):
     if array[i][3] != 0:
        Images.append(array[i][7])
        y.append(array[i][3])
        Images.append(array[i][8])
        y.append(array[i][3]+correction)
        Images.append(array[i][9])
        y.append(array[i][3]-correction)
        
#Function for opening, resizing and loading the images from path and selecting 
#the images with the help of list created
def loadImages(path):
    # return array of images
    loadedImages = []
    for i in range(len(Images)):
        img = PImage.open(path+Images[i])
        img = img.resize((200,96),PImage.ANTIALIAS)
        loadedImages.append(img)
    return loadedImages

path = "C:\\Users\\che\\CarND-Behavioral-Cloning-P3\\IMG\\"


# use the function created to load the images and see a sample of images
imgs = loadImages(path)

plt.subplot(321)
plt.imshow(imgs[0])

plt.subplot(322)
plt.imshow(imgs[1])

plt.subplot(323)
plt.imshow(imgs[2])

plt.subplot(324)
plt.imshow(imgs[3])

### create a numpy.ndarray of the pixels of the images loaded
image_in_pixels = []
for image in imgs:
   image = np.array(image,np.float32)
   image_in_pixels.append(image)
X = np.array(image_in_pixels,dtype = np.float32)

#cshuffle the data and create a test set
test_size = 0.2
seed=7
X,y = shuffle(X,y,random_state=seed)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,
                                                 random_state=seed)

#create a model with Normalization, cropping, convolution, 
#flatten, dense and dropout layers
model = Sequential()
model.add(Lambda(lambda X:(X/255.0)-0.5,input_shape=(96,200,3)))
model.add(Cropping2D(cropping=((25,5),(0,0)),input_shape=(96,200,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu',
                        input_shape=(66,200,3),border_mode='valid'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu',border_mode='valid'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu',border_mode='valid'))
model.add(Convolution2D(64,3,3,activation='relu',border_mode='valid'))
model.add(Convolution2D(64,3,3,activation='relu',border_mode='valid'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(1164,activation='relu',W_constraint=maxnorm(3)))
model.add(Dropout(0.3))
model.add(Dense(100,activation='relu',W_constraint=maxnorm(3))) 
model.add(Dropout(0.3)) 
model.add(Dense(50,activation='relu',W_constraint=maxnorm(3))) 
model.add(Dropout(0.3))         
model.add(Dense(10,activation='relu',W_constraint=maxnorm(3)))  
model.add(Dense(1,activation='linear'))

#set the number of epoches, learning rate, decay and momentum and 
#compile the model.select MSE as the metric.
epochs=40
lrate=0.01
decay=lrate/epochs
sgd=SGD(lr=lrate,decay=decay)
model.compile(loss="mean_squared_error",optimizer=sgd,momentum=0.9,nestrov=False,
              scoring=['mean_squared_error'])

#Checkpoint the best model
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss',verbose=1,
                             save_best_only=True,mode='min')
callbacks_list=[checkpoint]

#fix random seed for reproducibility and fit the model by creating a validation
#set and selecting a batch_size for weight updation
seed = 7
np.random.seed(seed)
history = model.fit(X_train,y_train,validation_split=0.2,nb_epoch=epochs,
                   batch_size=128,callbacks=callbacks_list, verbose=0)
#List data in history
print(history.history.keys())
#summarise history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

#load the best weight, compile model and make predictions on test set
model.load_weights("weights.best.hdf5")
model.compile(loss="mean_squared_error",optimizer=sgd,momentum=0.9,nestrov=False,
              scoring=['mean_squared_error'])
print("Created model and loaded weights from file")
#Calculate the testset MSE
score = mean_squared_error(y_test,model.predict(X_test))
print("score: %.3f%%" % (score))

#Save the model to file
model.save('model.h5')
print("model saved to disk")

