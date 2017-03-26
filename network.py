import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error

from dataset import *

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print('Epoch %05d: early stopping THR' % epoch)
            self.model.stop_training = True

def create_model(n_pre, n_post, feature_count, hidden_neurons):
    """ 
        creates, compiles and returns a RNN model 
        @param n_pre: the number of previous time steps (input)
        @param n_post: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    DROPOUT = 0.2
    LAYERS = 2
    
    model = Sequential()  
    model.add(LSTM(hidden_neurons, input_shape=(n_pre, feature_count), return_sequences=False))  
    model.add(Dropout(DROPOUT))
    model.add(RepeatVector(n_post))
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
    
    '''callbacks = [
        EarlyStoppingByLossVal(monitor='loss', value=0.001, verbose=1),
        ModelCheckpoint(kfold_weights_path, monitor='loss', save_best_only=True, verbose=0),
    ]'''
    
    history = model.fit(dataset.dataX, dataset.dataY, batch_size=2, nb_epoch=epoch_count, validation_split=0.05)
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model.save(model_folder + '/model.h5')
    print(history.history.keys())
    
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

    
def predict_multiple(model, dataset, steps_ahead):
    ''' 
        predicts multiple steps ahead
        @return predict array of shape (nb_samples, steps_ahead, nb_features)
    '''
    predict = np.empty((dataset.dataX.shape[0], steps_ahead, dataset.dataX.shape[2]))
    for sample in range(dataset.dataX.shape[0]):
        toPredict = dataset.dataX[sample:sample+1, :, :]
        for step in range(steps_ahead):
            predict[sample, step, :] = model.predict(toPredict)
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
        @param data_type: type of the data (string) for debug outputs
        arithmetic mean RMSE, std of RMSE
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

    return avg, std
    
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
