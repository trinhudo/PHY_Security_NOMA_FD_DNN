# Enhancing PHY-Security of FD-Enabled NOMA Systems Using Jamming and User Selection: Performance Analysis and DNN Evaluation

**Kyusung Shim, Tri Nhu Do, Toan-Van Nguyen, Daniel Benevides da Costa, and Beongku An**   

## Abstract

In this paper, we study the physical layer security (PHY-security) improvement method for a downlink non-orthogonal multiple access (NOMA) system in the presence of an active eavesdropper. To this end, we propose a full-duplex (FD)-enabled NOMA system and a promising scheme, called minimal transmitter selection (MTS) scheme, to support secure transmission. Specifically, the cell-center and cell-edge users act simultaneously as both receivers and jammers to degrade the eavesdropper channel condition. Additionally, the proposed MTS scheme opportunistically selects the transmitter to minimize the maximum eavesdropper channel capacity. To estimate the secrecy performance of the proposed methods, we derive an approximated closed-form expression for secrecy outage probability (SOP) and build a deep neural network (DNN) model for SOP evaluation. Numerical results reveal that the proposed NOMA system and MTS scheme improve not only the SOP but also the secrecy sum throughput. Furthermore, the estimated SOP through the DNN model is shown to be tightly close to other approaches, i.e., Monte-Carlo method and analytical expressions. The advantages and drawbacks of the proposed transmitter selection scheme are highlighted, along with insightful discussions.

## Key words

- Physical layer security (PHY-Security)
- Non-orthogonal multiple access (NOMA)
- Full-duplex (FD)
- Artificial noise (AN)
- Deep neural network (DNN)
- Secrecy outage probability (SOP)

# Demonstration of DNN-based Secrecy Outage Probability (SOP) Prediction
```python

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

NUM_EPOCHS = 30 # The number of epochs for training.
BATCH_SIZE = 200 #  

data = pd.read_csv('DataSet_FarUserMTS_1e5.csv')
data.shape
data.head()

x_in = data.drop(columns=['SOP']) 
# feature set
# get all data in each column except the column with title "SOP"

y_in = data['SOP'] 
# target variable (output)
# get data in the column with title "SOP"

x_train, x_test, y_train, y_test = train_test_split(x_in,y_in,
                                                    test_size = 0.1,
                                                    random_state = 0)

print(x_test)
print(y_test)

## build DNN model for regression
# Input Layer
model = keras.models.Sequential() # a basic feed-forward model
# Multiple Hidden Layer
model.add(keras.layers.Dense(128, activation='relu', 
                             input_shape = (12,), 
                             kernel_initializer='normal'))
model.add(keras.layers.Dense(128, activation='relu', 
                             kernel_initializer='normal'))
model.add(keras.layers.Dense(128, activation='relu', 
                             kernel_initializer='normal'))
model.add(keras.layers.Dense(128, activation='relu', 
                             kernel_initializer='normal'))
model.add(keras.layers.Dense(128, activation='relu', 
                             kernel_initializer='normal'))
# Output Layer
model.add(keras.layers.Dense(1, activation='linear',
                             kernel_initializer='normal'))  

# Setting for the DNN model training. 
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 1, factor = 0.8, min_lr = 1e-10) # for adaptive learning rate
optimizer = keras.optimizers.Adam(lr = 1e-2) # optimizer for update DNN model

model.compile(optimizer = optimizer,
              loss='mse',
              metrics=['mse'])

history = model.fit(x_train, y_train, 
                validation_split=0.1,
                epochs = NUM_EPOCHS, 
                batch_size = BATCH_SIZE,
                callbacks = [reduce_lr]
                )
```

           Unnamed: 0  Unnamed: 1  Unnamed: 2  ...  Unnamed: 9  Unnamed: 10  Unnamed: 11
    3582            4          35           5  ...         0.1          0.2          0.1
    60498           4          55          10  ...         0.2          0.2          0.2
    53227           4         -15           5  ...         0.2          0.1          0.1
    21333           4          40          10  ...         0.1          0.2          0.1
    3885            4          10          10  ...         0.1          0.2          0.2
    ...           ...         ...         ...  ...         ...          ...          ...
    89555           4           5           5  ...         0.1          0.1          0.2
    88135           4         -15           5  ...         0.1          0.1          0.2
    51888           4          60          10  ...         0.1          0.2          0.2
    51380           4           0          10  ...         0.1          0.1          0.1
    67294           4         -15           5  ...         0.1          0.1          0.2
    
    [10000 rows x 12 columns]
    3582     0.86520
    60498    0.99975
    53227    0.96625
    21333    0.80355
    3885     0.08580
              ...   
    89555    0.22685
    88135    0.99705
    51888    0.99650
    51380    0.84415
    67294    0.98175
    Name: SOP, Length: 10000, dtype: float64
    Epoch 1/30
    405/405 [==============================] - 3s 6ms/step - loss: 0.0770 - mse: 0.0770 - val_loss: 0.0020 - val_mse: 0.0020
    Epoch 2/30
    405/405 [==============================] - 2s 6ms/step - loss: 0.0015 - mse: 0.0015 - val_loss: 4.9305e-04 - val_mse: 4.9305e-04
    ...
    Epoch 29/30
    405/405 [==============================] - 2s 6ms/step - loss: 1.0618e-05 - mse: 1.0618e-05 - val_loss: 1.0652e-05 - val_mse: 1.0652e-05
    Epoch 30/30
    405/405 [==============================] - 2s 6ms/step - loss: 1.0404e-05 - mse: 1.0404e-05 - val_loss: 1.0470e-05 - val_mse: 1.0470e-05

```python
# plt.plot(history.history['mean_squared_error'])
# plt.plot(history.history['val_mean_squared_error'])

# If the above history plot does not work, pls use the following
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])

plt.yscale('log')
plt.legend(loc='upper right')
plt.grid(True)
plt.ylabel('Estimator MSE')
plt.xlabel('Number of Epochs')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
```

![png](output_2_1.png)


```python
# the saved trainined DNN model
print(model.summary())
model.save('Trained_DNN_FarUserMTS.h5')
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_6 (Dense)              (None, 128)               1664      
    _________________________________________________________________
    dense_7 (Dense)              (None, 128)               16512     
    _________________________________________________________________
    dense_8 (Dense)              (None, 128)               16512     
    _________________________________________________________________
    dense_9 (Dense)              (None, 128)               16512     
    _________________________________________________________________
    dense_10 (Dense)             (None, 128)               16512     
    _________________________________________________________________
    dense_11 (Dense)             (None, 1)                 129       
    =================================================================
    Total params: 67,841
    Trainable params: 67,841
    Non-trainable params: 0
    _________________________________________________________________
    None



```python
# verify the trained model
new_model = keras.models.load_model('Trained_DNN_FarUserMTS.h5')
y_pred = new_model.predict(x_test)

print('===============================================')
print('Root Mean Squared Error:', 
      np.sqrt(mean_squared_error(y_test, y_pred))) 
print('===============================================')
print('END')
```

    ===============================================
    Root Mean Squared Error: 0.00331421030026788
    ===============================================
    END
