import numpy as np
np.random.seed(123)  # for reproducibility

import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from attrdict import AttrDict
import pylab as plt

from dataset import *
 
#TRAINING_PATH = '/media/isa/VIS1/data/total_2mtemp.grib'
TRAINING_PATH = 'soil_temp1_summer05.grib'

def example():
    window_size = 5
    hidden_neurons = 100
    epoch_count = 10
    params = AttrDict()
    params.window_size = window_size
    params.start_lat = 47.3769
    params.end_lat = 47.3769
    params.start_lon = 8.5417
    params.end_lon = 9.0
    params.grib_file = TRAINING_PATH
    params.max_frames = 1000
    params.test_fraction = 0.3

    # load the data from the .grib files
    data = DatasetMultiple(params)
    
    # create and fit the LSTM network
    model = Sequential()  
    model.add(LSTM(hidden_neurons, input_shape=(window_size, data.vector_size), return_sequences=False))  
    model.add(Dense(data.vector_size))  
    #model.add(Activation("linear"))  
    model.compile(loss="mean_squared_error", optimizer="rmsprop")  
    model.fit(data.trainX, data.trainY, batch_size=1, nb_epoch=epoch_count, validation_split=0.05)
    
    # make predictions
    trainPredict = model.predict(data.trainX)
    testPredict = model.predict(data.testX)
    
    # invert predictions
    trainPredict = data.scaler.inverse_transform(trainPredict)
    data.trainY = data.scaler.inverse_transform(data.trainY)
    testPredict = data.scaler.inverse_transform(testPredict)
    data.testY = data.scaler.inverse_transform(data.testY)
    
    # calculate root mean squared error
    #trainScore = math.sqrt(mean_squared_error(data.trainY[0,:], trainPredict[:,0]))
    #print('Train Score: %.2f RMSE' % (trainScore))
    #testScore = math.sqrt(mean_squared_error(data.testY[0,:], testPredict[:,0]))
    #print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(data.frames)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[window_size:len(trainPredict)+window_size, :] = trainPredict[:,:]
    
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(data.frames)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(window_size*2)+1:len(data.frames)-1, :] = testPredict[:,:]
    
    # plot baseline and predictions
    plt.plot(data.scaler.inverse_transform(data.frames))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

def main():
    example()
    return 1

if __name__ == "__main__":
    sys.exit(main())
