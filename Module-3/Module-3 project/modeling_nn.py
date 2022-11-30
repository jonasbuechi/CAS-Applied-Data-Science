# load modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from time import sleep
from IPython.display import clear_output
from collections import Counter


#Define wind force class names
wind_force_class_names = ["No wind","Weak wind force","Middle wind force","Strong wind force"]


# fun to crop img
def fn_crop_image(img_array, y_start, x_start):
    y_heigth = 60
    x_width = 110
    
    crop_image = img_array[y_start:y_start+y_heigth,x_start:x_start+x_width] # set RGB to 0 to only analyze red chanel
    return crop_image

# fun to resize img
# to do if further dimension reduction is desired
    

# define data directory
DATADIR = "../data/Originals/"

# import labels
label_df = pd.read_csv("../data/labels_old_camera.csv")
label_df = label_df.rename(columns={"Unnamed: 0": "img"})

# retain only labels / img with meaningful label
label_df = label_df[label_df["wind_force"] != "0"]

# array with img names of labeled img
labeled_img = label_df["img"].to_numpy()

# create numeric wind force variable
label_df["wind_force_num"] = np.where(label_df["wind_force"] == "n", 0,
                             np.where(label_df["wind_force"] == "w", 1,
                             np.where(label_df["wind_force"] == "m", 2,
                             np.where(label_df["wind_force"] == "s", 3, -1))))



#Create Training Data
training_data = []

#Set True to harmonize amount of pictures per label (multiply pictures and labels)
harmonize = False
# Get count per label value and calculate factors for harmonizing amounts
label_value_counts = label_df["wind_force_num"].value_counts()

maxLabels = label_value_counts.max()
factor0 = round(maxLabels / label_value_counts.loc[0])
factor1 = round(maxLabels / label_value_counts.loc[1])
factor2 = round(maxLabels / label_value_counts.loc[2])
factor3 = round(maxLabels / label_value_counts.loc[3])
if harmonize:
  print(f"factors for 0,1,2,3 category: {factor0,factor1,factor2,factor3}")


def create_training_data():
    for img in labeled_img:
        path = os.path.join(DATADIR, img)
        img_array = cv2.imread(os.path.join(path))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # from BGR to RGB
        new_array = fn_crop_image(img_array, 250, 570)
        wind_label = label_df.loc[label_df["img"] == img, "wind_force_num"] # get label to coresponding img
        wind_label = np.ndarray.item(wind_label.to_numpy()) # convert to single scalar integer
        if harmonize:
          if wind_label == 0: #No wind
            for i in range(factor0):
              training_data.append([new_array, wind_label])
          if wind_label == 1: #Middle wind
            for i in range(factor1):
              training_data.append([new_array, wind_label])
          if wind_label == 2: #Middle wind
            for i in range(factor2):
              training_data.append([new_array, wind_label])
          if wind_label == 3: #Strong wind
            for i in range(factor3):
              training_data.append([new_array, wind_label])
        else:
          training_data.append([new_array, wind_label])
    
create_training_data()


images = []
labels = []

for feature, label in training_data:
    images.append(feature)
    labels.append(label)

# mutate to np.array
np_images = np.array(images).reshape(-1, np.array(images).shape[1], np.array(images).shape[2], np.array(images).shape[3])
np_labels = np.array(labels)


from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.inspection import permutation_importance
from joblib import dump

# common visualization module
#import plotly.express as px
#import seaborn as sns
#sns.set()
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from time import time as timer
import tarfile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import visualkeras


from matplotlib import animation
from IPython.display import HTML


#Convert image shape and split to train and test set
label_amount = np_labels.size

print(f"np_images shape: {np_images.shape}")
np_images_red = np_images[:,:,:,0] #reduce to red channel
print(f"np_images_red shape: {np_images_red.shape}")

np_images_red_flatten = np_images_red[:label_amount,:].reshape(label_amount,-1)
print(f"np_images_red_flatten shape: {np_images_red_flatten.shape}")

#1) Split into train and test set: HEADS-UP: RANDOM_STATE set to 7 for reproducable results..
#x_train, x_test, y_train, y_test = train_test_split(np_images_red_flatten, np_labels, test_size=0.2,random_state=7)
x_train, x_test, y_train, y_test = train_test_split(np_images_red, np_labels, test_size=0.2,random_state=7) # da im NN geflatted wird
X_full, y_full = np_images_red, np_labels

print(f"np_labels shape: {np_labels.shape}")


# standardize pixel-values
x_train = x_train/255
x_test = x_test/255

X_full = X_full/255


# ## Model Training
# ### Simple fully connected Neurnal Network

# Model 1: Simple fully connected (2 Layer)

model1 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(60, 110)),
  tf.keras.layers.Dense(4, activation='softmax')
])

model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model1.summary()


save_path = 'save/mnist_{epoch}.ckpt'
save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True)

hist = model1.fit(x=x_train, y=y_train,
          epochs=500, batch_size=64, 
          validation_data=(x_test, y_test))


fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['accuracy'])
axs[1].plot(hist.epoch, hist.history['val_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()


# Model fit with batch_size=64
#model1.fit(x=x_train, y=y_train,
#          epochs=500, batch_size=64, 
#          validation_data=(x_test, y_test))

model1.evaluate(x_test,  y_test, verbose=2)

predict_x=model1.predict(x_test) 
y_pred=np.argmax(predict_x,axis=1)

#Generate the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

import seaborn as sns

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues_r')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(["No wind","Weak wind","Middle wind","Strong wind"])
ax.yaxis.set_ticklabels(["No wind","Weak wind","Middle wind","Strong wind"])

## Display the visualization of the Confusion Matrix.
plt.show()


model1.save('../data/model1_nn_bs64_ep500_noharmon')



# Model with automatical valdiation split -> peforms significantly better, but is probably overfit
model1.fit(x=X_full, y=y_full,
          epochs=100, batch_size=64, 
          validation_split=0.1)

model1.evaluate(x_test,  y_test, verbose=2)

model2 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(60, 110)),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(4, activation='softmax')
])


model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model2.summary()


model2 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(60, 110)),
  tf.keras.layers.Dense(1024, activation='relu'),
  #tf.keras.layers.Dense(256, activation='relu'),
  #tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(4, activation='softmax')
])


model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model2.summary()


# Model fit with batch_size=64
model2.fit(x=x_train, y=y_train,
          epochs=500, batch_size=64, 
          validation_data=(x_test, y_test))

model2.evaluate(x_test,  y_test, verbose=2)


model2.evaluate(x_test,  y_test, verbose=2)


# ## Visualisierung Models


save_path = 'save/mnist_{epoch}.ckpt'
save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_weights_only=True)

hist = model2.fit(x=x_train, y=y_train,
          epochs=500, batch_size=64, 
          validation_data=(x_test, y_test))




fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(hist.epoch, hist.history['loss'])
axs[0].plot(hist.epoch, hist.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(hist.epoch, hist.history['accuracy'])
axs[1].plot(hist.epoch, hist.history['val_accuracy'])

axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()


model1.evaluate(x_test,  y_test, verbose=2)


im_id = 0
y_pred = model1(x_test)

y_pred_most_probable = np.argmax(y_pred[im_id])
print('true lablel: ', y_test[im_id],
      '; predicted: ',  y_pred_most_probable,
      f'({wind_force_class_names[y_pred_most_probable]})')
plt.imshow(x_test[im_id], cmap='gray');


# Safe logistic regression model to disk with joblib
from joblib import dump

nn_filename = "model1_nn_bs64_ep500_harmon.joblib"
dump(model1, "../data/"+nn_filename) 



