import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error

from dataset import *

def create_model(steps_before, steps_after, feature_count, hidden_neurons):
    """ 
        creates, compiles and returns a RNN model 
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    DROPOUT = 0.5
    LAYERS = 2
    
    model = Sequential()  
    model.add(LSTM(hidden_neurons, input_shape=(steps_before, feature_count)))  
    model.add(Dropout(DROPOUT))
    model.add(RepeatVector(steps_after))
    for _ in range(LAYERS):
        model.add(LSTM(hidden_neurons, return_sequences=True))
        model.add(Dropout(DROPOUT))

    model.add(TimeDistributed(Dense(feature_count)))
    model.add(Activation('sigmoid'))   
    
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])  
    return model

def train_model(model, dataset, epoch_count, model_folder):
    """ 
        trains a model given the dataset to train on
        @param model: the model to train
        @param dataset: the dataset to train the model on
        @param epoch_count: number of epochs to train
        @param model_folder: the trained model as well as plots for the training history are saved there
        
        TODO: maybe specify if the model needs to be saved between epochs?
        TODO: maybe specify a target validation loss? (monitor='val_loss')
    """
    history = model.fit(dataset.dataX, dataset.dataY, batch_size=2, nb_epoch=epoch_count, validation_split=0.05)
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model.save(model_folder + '/model.h5')

    # plot training and val loss/accuracy
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_folder + '/history_acc.png')
    plt.cla()    

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(model_folder + '/history_loss.png')
    plt.cla()

def predict_multiple(model, dataset, steps_after):
    ''' 
        predicts multiple steps ahead
        @return predict array of shape (nb_samples, steps_after, nb_features)
    '''
    predict = np.empty((dataset.dataX.shape[0], steps_after, dataset.dataX.shape[2]))
    for sample in range(dataset.dataX.shape[0]):
        toPredict = dataset.dataX[sample:sample+1, :, :]
        for step in range(steps_after):
            predict[sample, step:step+1, :] = model.predict(toPredict)
            toPredict = np.concatenate((toPredict[:, 1:, :], predict[sample:sample+1, step:step+1, :]), axis=1)

    return predict
    
def evaluate_model(model, dataset, data_type=''):
    """ 
        evaluates the model given the dataset (training or test data)
        @param model: the model to evaluate_model
        @param dataset: data to evaluate
        @param data_type: type of the data (string) for debug outputs
        @return the predicted vector
    """
    # make predictions
    predict = model.predict(dataset.dataX)
   
    # invert predictions
    predict = dataset.scaler.inverse_transform(predict)
    dataY = dataset.scaler.inverse_transform(dataset.dataY)
  
    # calculate root mean squared error
    scores = np.empty((dataY.shape[1]))
    for i in range(dataY.shape[1]):
        scores[i] = math.sqrt(mean_squared_error(dataY[:,i], predict[:,i]))
    
    avg = np.mean(scores)
    std = np.std(scores)
    print(data_type + ' Score mean: %.2f RMSE' % avg)    
    print(data_type + ' Score std: %.2f RMSE' % std) 

    return predict
    
def evaluate_model_score(model, dataset):
    """ 
        evaluates the model given the dataset and returns
        the avg mean squared error.
        @param model: the model to evaluate_model
        @param dataset: data to evaluate
        @return arithmetic mean RMSE, std of RMSE, each one per timestep and per feature, shape is (timestep, feature, [0=rmse, 1=mad])
    """
    # make predictions
    predict = dataset.predict_data(model)
    _, dataY = dataset.inverse_transform_data()

    return evaluate_model_score_raw(dataY, predict)
    
def evaluate_model_score_raw(dataY, predict):
    """ 
        evaluates the model given the dataset and the prediction (both already scaled)
        @param dataY: dataY from the dataset
        @param predict: already predicted data
        @return arithmetic mean RMSE, std of RMSE, each one per timestep and per feature, shape is (timestep, feature, [0=rmse, 1=mad])
    """
    # calculate root mean squared error and
    # mean absolute deviation
    # scores are (timestep, feature, [0=rmse, 1=mad])
    scores = np.empty((dataY.shape[1], dataY.shape[2], 2))
    for i in range(dataY.shape[1]): # loop over timesteps
        for j in range(dataY.shape[2]): # loop over features
            scores[i, j, 0] = math.sqrt(mean_squared_error(dataY[:,i,j], predict[:,i,j]))
            errors = np.absolute(dataY[:,i,j] - predict[:,i,j])
            scores[i, j, 0] = np.mean(np.absolute(errors - np.mean(errors, axis=0)), axis=0)

    return scores
    
def evaluate_model_score_2(model, dataset):
    """ 
        evaluates the model given the dataset and returns
        the avg mean squared error of the first feature and then all features
        @param model: the model to evaluate_model
        @param dataset: data to evaluate
        @param data_type: type of the data (string) for debug outputs
    """
    # make predictions
    predict = model.predict(dataset.dataX)
   
    # invert predictions
    predict = dataset.scaler.inverse_transform(predict)
    dataY = dataset.scaler.inverse_transform(dataset.dataY)
 
    # calculate root mean squared error and

    scores = []
    for i in range(dataY.shape[1]):
        scores.append(math.sqrt(mean_squared_error(dataY[:,i], predict[:,i])))
    
    avg_1 = scores[0]
    avg_all = sum(scores)/len(scores)
    print('Score average 1: %.2f RMSE' % avg_1)    
    print('Score average all: %.2f RMSE' % avg_all)  
    
    return avg_1, avg_all
