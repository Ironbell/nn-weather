import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout  
from sklearn.metrics import mean_squared_error

from dataset import *

def create_model(window_size, feature_count):
    """ 
        creates, compiles and returns a RNN model 
        @param window_size: the number of previous time steps
        @param feature_count: the number of features in the model
    """
    hidden_neurons = 200
    dropout = 0.2
    
    model = Sequential()  
    model.add(LSTM(hidden_neurons, input_shape=(window_size, feature_count), return_sequences=False))  
    model.add(Dropout(dropout))
    #model.add(LSTM(hidden_neurons, return_sequences=False))
    #model.add(Dropout(dropout))
    model.add(Dense(feature_count))  
    model.add(Activation("linear"))   
    
    model.compile(loss="mean_squared_error", optimizer="rmsprop")  
    return model

def train_model(model, dataset, epoch_count):
    """ 
        trains a model given the dataset to train on
        @param model: the model to train
        @param dataset: the dataset to train the model on
        @param epoch_count: number of epochs to train
        
        TODO: maybe specify if the model needs to be saved between epochs?
        TODO: maybe specify a target validation loss?
    """
    model.fit(dataset.dataX, dataset.dataY, batch_size=1, nb_epoch=epoch_count, validation_split=0.05)
    
def evaluate_model(model, dataset, data_type=''):
    """ 
        evaluates the model given the dataset (training or test data)
        @param model: the model to evaluate_model
        @param dataset: data to evaluate
        @param data_type: type of the data (string) for debug outputs
    """
    # make predictions
    predict = model.predict(dataset.dataX)
   
    # invert predictions
    predict = dataset.scaler.inverse_transform(predict)
    dataset.dataY = dataset.scaler.inverse_transform(dataset.dataY)
    
    for i in range(dataset.dataX.shape[0]):
        dataset.dataX[i] = dataset.scaler.inverse_transform(dataset.dataX[i])
  
    # calculate root mean squared error
    scores = []
    for i in range(dataset.dataY.shape[1]):
        scores.append(math.sqrt(mean_squared_error(dataset.dataY[:,i], predict[:,i])))
    
    avg = sum(scores)/len(scores)
    print(data_type + ' Score average: %.2f RMSE' % avg)    

    return predict
    
def evaluate_model_score(model, dataset):
    """ 
        evaluates the model given the dataset and returns
        the avg mean squared error.
        @param model: the model to evaluate_model
        @param dataset: data to evaluate
        @param data_type: type of the data (string) for debug outputs
    """
    # make predictions
    predict = model.predict(dataset.dataX)
   
    # invert predictions
    predict = dataset.scaler.inverse_transform(predict)
    dataset.dataY = dataset.scaler.inverse_transform(dataset.dataY)
    
    for i in range(dataset.dataX.shape[0]):
        dataset.dataX[i] = dataset.scaler.inverse_transform(dataset.dataX[i])
  
    # calculate root mean squared error
    scores = []
    for i in range(dataset.dataY.shape[1]):
        scores.append(math.sqrt(mean_squared_error(dataset.dataY[:,i], predict[:,i])))
    
    avg = sum(scores)/len(scores)
    print('Score average: %.2f RMSE' % avg)    

    return avg
