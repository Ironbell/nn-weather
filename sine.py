'''Trains a simple RNN-LSTM on sine waveform data.
'''
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import seed, random

np.random.seed(1337)  # for reproducibility

# Simulate data
dates = pd.date_range(start='2009-01-01', end='2015-12-31', freq='D')
n = len(dates)
a = np.sin(np.arange(n) * 2 * np.pi / 7)
# b = np.sin(np.arange(n) * 2 * np.pi / 7)
# c = np.sin(np.arange(n) * 2 * np.pi / 7)
# pdata = pd.DataFrame({"a":a, "b":b, "c":c})
pdata = pd.DataFrame({"a":a}, index=dates)
data = pdata

# visualize data
n_plot = 100
plt.figure()
plt.plot(range(1, n_plot+1), data.a[:n_plot])
plt.xlabel('Index')
plt.ylabel('Value')
plt.tight_layout()
plt.show()

def _load_data(data, n_prev = 100, n_post = 10):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev-n_post):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev:i+n_prev+n_post].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    
    print('X shape')
    print(alsX.shape)
    print('Y shape')
    print(alsY.shape)

    return alsX, alsY

def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(df) * (1 - test_size)))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])
    return (X_train, y_train), (X_test, y_test)

# retrieve data
(X_train, y_train), (X_test, y_test) = train_test_split(data)

# define model structure
in_out_neurons = 1
hidden_neurons = 300
model = Sequential()
model.add(LSTM(input_dim=in_out_neurons, output_dim=hidden_neurons, return_sequences=False))
model.add(Dense(output_dim=in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
model.fit(X_train, y_train, batch_size=50, nb_epoch=10, validation_split=0.05)

# evaluate model fit
score = model.evaluate(X_test, y_test)
print('Test score:', score)

# visualize predictions
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

_, axarr = plt.subplots(5, sharex=True, sharey=True)
axarr[0].plot(a[(len(a) - n_plot):len(a)])
axarr[0].set_title('Test Population')
axarr[1].plot(y_test[:n_plot])
axarr[1].set_title('Test Observation')
axarr[2].plot(test_prediction[:n_plot])
axarr[2].set_title('Prediction')
axarr[3].plot(y_test[:n_plot])
axarr[3].plot(test_prediction[:n_plot])
axarr[3].set_title('Test Observation and Prediction')
axarr[4].plot(a[(len(a) - n_plot):len(a)])
axarr[4].plot(test_prediction[:n_plot])
axarr[4].set_title('Test Population and Prediction')
plt.show()



