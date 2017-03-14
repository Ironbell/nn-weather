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
    hidden_neurons = 100
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
    model.fit(dataset.trainX, dataset.trainY, batch_size=1, nb_epoch=epoch_count, validation_split=0.05)
    
def evaluate_model(model, dataset):
    """ 
        evaluates the model given the dataset (training and test data)
        @param model: the model to evaluate_model
        @param dataset: training and test data to evaluate
    """
    # make predictions
    trainPredict = model.predict(dataset.trainX)
    testPredict = model.predict(dataset.testX)
    
    # invert predictions
    trainPredict = dataset.scaler.inverse_transform(trainPredict)
    dataset.trainY = dataset.scaler.inverse_transform(dataset.trainY)
    testPredict = dataset.scaler.inverse_transform(testPredict)
    dataset.testY = dataset.scaler.inverse_transform(dataset.testY)
    
    for i in range(dataset.trainX.shape[0]):
        dataset.trainX[i] = dataset.scaler.inverse_transform(dataset.trainX[i])
    for i in range(dataset.testX.shape[0]):
        dataset.testX[i] = dataset.scaler.inverse_transform(dataset.testX[i])
    
    # calculate root mean squared error
    scores = []
    for i in range(dataset.trainY.shape[1]):
        scores.append(math.sqrt(mean_squared_error(dataset.trainY[:,i], trainPredict[:,i])))
    
    avg = sum(scores)/len(scores)
    print('Train Score average: %.2f RMSE' % avg)
        
    scores = []
    for i in range(dataset.testY.shape[1]):
        scores.append(math.sqrt(mean_squared_error(dataset.testY[:,i], testPredict[:,i])))

    avg = sum(scores)/len(scores)
    print('Test Score average: %.2f RMSE' % avg)    

    return trainPredict, testPredict
