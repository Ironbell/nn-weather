import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
import json, attrdict

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, TimeDistributed, Flatten, RepeatVector
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error

from dataset import *

def get_default_model_params():
    '''
        returns the default params for a model.
    '''
    params = AttrDict()
    params.lstm_layers = 1
    params.dropout = 0.2
    params.conv_filters = 8
    params.conv_filter_size = 2
    params.lstm_neurons = 64
    params.dense_neurons = 64
    params.activation = 'linear'
    return params

def create_model(model_params, data_params):
    """ 
        creates, compiles and returns a CNN + RNN model 
        @param model_params: parameters for the model
        @param data_params: parameters for the dataset
    """
    diameter = 1 + 2 * data_params.radius
    nb_features = len(data_params.grib_parameters)

    model = Sequential()  
    model.add(TimeDistributed(Conv2D(filters=model_params.conv_filters, kernel_size=model_params.conv_filter_size, padding='same', input_shape=(diameter, diameter, nb_features)), input_shape=(data_params.steps_before, diameter, diameter, nb_features)))
    
    if (diameter > 1):
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
  
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(model_params.dense_neurons)))

    for i in range(model_params.lstm_layers):
        model.add(LSTM(model_params.lstm_neurons, return_sequences=(i == model_params.lstm_layers)))
        model.add(Dropout(model_params.dropout))
        
    model.add(RepeatVector(data_params.steps_after))
    
    model.add(LSTM(model_params.lstm_neurons, return_sequences=True))

    model.add(TimeDistributed(Dense(nb_features)))
    model.add(TimeDistributed(Activation(model_params.activation)))  
    
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])  
    return model

def train_model(model, dataset, epoch_count, model_folder):
    """ 
        trains a model given the dataset to train on
        @param model: the model to train
        @param dataset: the dataset to train the model on
        @param epoch_count: number of epochs to train
        @param model_folder: the trained model as well as plots for the training history are saved there
    """
    history = model.fit(dataset.dataX, dataset.dataY, batch_size=2, epochs=epoch_count, validation_split=0.05)
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # save model
    model.save(model_folder + '/model.h5')
    # save dataset parameters as json
    with open(model_folder + '/params.json', 'w') as fp:
        json.dump(dataset.params, fp)

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
    
    plot_model(model, to_file=model_folder + '/model.png')

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
        @return mean absolute error, std of absolute error each one per timestep and per feature, shape is (feature, [0=mean, 1=std])
    """
    scores = np.empty((dataY.shape[1], dataY.shape[2], 2))
    for i in range(dataY.shape[1]): # loop over timesteps
        for j in range(dataY.shape[2]): # loop over features
            ad = np.absolute(dataY[:,i,j] - predict[:,i,j])
            scores[i, j, 0] = np.mean(ad)
            scores[i, j, 1] = np.std(ad)

    return scores

def evaluate_model_score_compare(dataY, predict):
    '''
        scores are (timestep, feature, [0=mean(squared distance), 1=std(squared distance), 3=mean(absolute distance), 4=std(absolute distance))
        
        #d=(x-y).^2;   % SD   (squared distance)
        #m=mean(d)     % MSD  (mean squared distance, aka. mean squared error MSE)
        #s=std(d)

        #d=abs(x-y);   % AD   (absolute distance)
        #m=mean(d)     % MAD  (mean absolute distance)
        #s=std(d)
    '''
    
    scores = np.empty((dataY.shape[1], dataY.shape[2], 5))
    for i in range(dataY.shape[1]): # loop over timesteps
        for j in range(dataY.shape[2]): # loop over features
            scores[i, j, 4] = math.sqrt(mean_squared_error(dataY[:,i,j], predict[:,i,j]))
            
            sd = np.square(dataY[:,i,j] - predict[:,i,j])
            scores[i, j, 0] = np.mean(sd)
            scores[i, j, 1] = np.std(sd)
            
            ad = np.absolute(dataY[:,i,j] - predict[:,i,j])
            scores[i, j, 2] = np.mean(ad)
            scores[i, j, 3] = np.std(ad)

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
 
    # calculate root mean squared error
    scores = []
    for i in range(dataY.shape[1]):
        scores.append(math.sqrt(mean_squared_error(dataY[:,i], predict[:,i])))
    
    avg_1 = scores[0]
    avg_all = sum(scores)/len(scores)
    print('Score average 1: %.2f RMSE' % avg_1)    
    print('Score average all: %.2f RMSE' % avg_all)  
    
    return avg_1, avg_all
