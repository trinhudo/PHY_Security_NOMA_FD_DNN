
# call packet for machine learning model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ReduceLROnPlateau
#from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D

data = pd.read_csv('DataSet_FarUserMTS_1e5.csv')
data.shape
data.head()
x_in = data.drop(columns=['SOP']) # get all data in each column except the column with title "cap"
y_in = data['SOP'] # get data in the column with title "cap"

x_train, x_test, y_train, y_test = train_test_split(x_in,y_in,
                                                    test_size = 0.1,
                                                 random_state = 0)

print(x_test)
print(y_test)

## build DNN model fro regression
# Input Layer
model = keras.models.Sequential() # a basic feed-forward model
# Multiple Hidden Layer
model.add(keras.layers.Dense(128, activation='relu', input_shape = (12,), kernel_initializer='normal'))
model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='normal'))
model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='normal'))
model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='normal'))
model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='normal'))
# Output Layer
model.add(keras.layers.Dense(1, activation='linear',kernel_initializer='normal'))  # our output layer. 


# Setting for the DNN model training. 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 1, factor = 0.8, min_lr = 1e-10) # for adaptive learning rate
optimizer = keras.optimizers.Adam(lr = 1e-2) # optimizer for update DNN model
# 
Epoch = 30 # The number of round for training.
BATCH_SIZE = 200 #  
#

model.compile(optimizer = optimizer,
              loss='mse',
              metrics=['mse'])

history = model.fit(x_train, y_train, 
                validation_split=0.1,
                epochs = Epoch, 
                batch_size = BATCH_SIZE,
                callbacks = [reduce_lr]
                )


plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.yscale('log')
plt.legend(loc='upper right')
plt.grid(True)
plt.ylabel('Estimator MSE')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# the saved trainined DNN model
print(model.summary())
model.save('Trained_DNN_FarUserMTS.h5')

# verify the trained model
new_model = keras.models.load_model('Trained_DNN_FarUserMTS.h5')
y_pred = new_model.predict(x_test)



print('===============================================')
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred))) 
print('===============================================')
print('END')
