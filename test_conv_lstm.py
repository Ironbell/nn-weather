import os, json
from attrdict import AttrDict
from keras.models import load_model
import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, TimeDistributed, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.metrics import mean_squared_error

from dataset import *

EPOCHS = 200
#GRIB_FOLDER = '/home/isa/sftp/'
GRIB_FOLDER = '/media/isa/VIS1/'

def create_model(steps_before, radius, nb_features):
    """ 
        creates, compiles and returns a RNN model 
        @param steps_before: the number of previous time steps (input). 
        @param radius: the radius of the square around the feature of interest
    """
    DROPOUT = 0.2
    HIDDEN_NEURONS = 128
    diameter = 1 + 2 * radius

    model = Sequential()
    model.add(TimeDistributed(Conv2D(filters=8, kernel_size=(2,2), padding='same', input_shape=(diameter, diameter, nb_features)), input_shape=(steps_before, diameter, diameter, nb_features)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    #model.add(TimeDistributed(Dropout(DROPOUT)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(HIDDEN_NEURONS)))
    model.add(LSTM(HIDDEN_NEURONS, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(HIDDEN_NEURONS, return_sequences=False))
    model.add(Dense(nb_features))
    
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
    
    scores = np.empty((dataY.shape[1], 5))
   
    for i in range(dataY.shape[1]):
        scores[i, 4] = math.sqrt(mean_squared_error(dataY[:, i], predict[:, i]))
        
        sd = np.square(dataY[:, i] - predict[:, i])
        scores[i, 0] = np.mean(sd)
        scores[i, 1] = np.std(sd)
        
        ad = np.absolute(dataY[:, i] - predict[:, i])
        scores[i, 2] = np.mean(ad)
        scores[i, 3] = np.std(ad)

    return scores

def test_model():
    ''' 
        train the network
    '''
    train_params = AttrDict()
    train_params.steps_before = 20
    train_params.forecast_distance = 0
    train_params.steps_after = 1
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.radius = 2
    train_params.grib_folder = GRIB_FOLDER
    train_params.grib_parameters = ['temperature']
    train_params.months = [1,2,3,4,5,6,7,8,9,10,11,12]
    train_params.years = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]
    train_params.hours = [0, 6, 12, 18]

    # train and save the model files
    print ('training started...')
    model_folder = 'test_conv_lstm/model'
    trainData = DatasetSquareArea(train_params)
        
    # create and fit the LSTM network
    print('creating model...')
    model = create_model(train_params.steps_before, train_params.radius, len(train_params.grib_parameters))
    train_model(model, trainData, EPOCHS, model_folder)
    
def evaluate_constant_baseline():
    ''' 
        evaluation of the constant baseline
    '''
    radius = 2
    train_params = AttrDict()
    train_params.steps_before = 20
    train_params.forecast_distance = 0
    train_params.steps_after = 1
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.radius = radius
    train_params.grib_folder = GRIB_FOLDER
    train_params.grib_parameters = ['temperature']
    train_params.hours = [0, 6, 12, 18]

    # evaluate on whole 2000 and save results
    train_params.years = [2000]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]

    for month in months:
        train_params.months = [month]
        testData = DatasetSquareArea(train_params)
        
        subfolder = 'test_conv_lstm/constant_baseline/month_' + str(month)
        
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        
        predict = testData.predict_constant(flatten=False)
        dataX, dataY = testData.inverse_transform_data(flatten=False)
        np.save(subfolder + '/prediction.npy', predict)
        
        score = evaluate_model_score_compare(dataY, predict) 
        np.save(subfolder + '/score.npy', score)
        print('score shape:')
        print(score.shape)

        # plot the first results as a sanity check
        nb_results = min(10, predict.shape[0])
        
        nan_array = np.empty((testData.params.steps_before - 1))
        nan_array.fill(np.nan)
        nan_array2 = np.empty(testData.params.steps_after)
        nan_array2.fill(np.nan)
        ind = np.arange(testData.params.steps_before + testData.params.steps_after)

        for i in range(nb_results):
            start_date = testData.frames_data[testData.frames_idx[i + testData.params.steps_before]]
            end_date = testData.frames_data[testData.frames_idx[i + testData.params.steps_before + testData.params.steps_after - 1]]
            
            start_date_s = start_date.date + '-' + start_date.time
            end_date_s = end_date.date + '-' + end_date.time

            # plot temperature forecasts
            fig, ax = plt.subplots()

            forecasts = np.concatenate((nan_array, dataX[i, -1:, radius, radius, 0], predict[i:i+1, 0]))
            ground_truth = np.concatenate((nan_array, dataX[i, -1:, radius, radius, 0], dataY[i:i+1, 0]))
            network_input = np.concatenate((dataX[i, :, radius, radius, 0], nan_array2))
         
            ax.plot(ind, network_input, 'b-x', label='Network input')
            ax.plot(ind, forecasts, 'r-x', label='Constant forecast')
            ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')
            
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            plt.title('Many to One Forecast (' + start_date_s + ' -- ' + end_date_s + ')')
            plt.legend(loc='best')
            plt.savefig(subfolder + '/plot_' + str(i) + '.png')
            plt.cla()
            
def evaluate_model():
    ''' 
        evaluation of the model
    '''
    radius = 2
    train_params = AttrDict()
    train_params.steps_before = 20
    train_params.forecast_distance = 0
    train_params.steps_after = 1
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.radius = radius
    train_params.grib_folder = GRIB_FOLDER
    train_params.grib_parameters = ['temperature']
    train_params.hours = [0, 6, 12, 18]

    # evaluate on whole 2000 and save results
    train_params.years = [2000]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    model = load_model('test_conv_lstm/model/model.h5')

    for month in months:
        train_params.months = [month]
        testData = DatasetSquareArea(train_params)
        
        subfolder = 'test_conv_lstm/temperature/month_' + str(month)
        
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        
        predict = testData.predict_data(model, flatten=False)
        dataX, dataY = testData.inverse_transform_data(flatten=False)
        np.save(subfolder + '/prediction.npy', predict)
        
        score = evaluate_model_score_compare(dataY, predict) 
        np.save(subfolder + '/score.npy', score)
        print('score shape:')
        print(score.shape)

        # plot the first results as a sanity check
        nb_results = min(10, predict.shape[0])
        
        nan_array = np.empty((testData.params.steps_before - 1))
        nan_array.fill(np.nan)
        nan_array2 = np.empty(testData.params.steps_after)
        nan_array2.fill(np.nan)
        ind = np.arange(testData.params.steps_before + testData.params.steps_after)

        for i in range(nb_results):
            start_date = testData.frames_data[testData.frames_idx[i + testData.params.steps_before]]
            end_date = testData.frames_data[testData.frames_idx[i + testData.params.steps_before + testData.params.steps_after - 1]]
            
            start_date_s = start_date.date + '-' + start_date.time
            end_date_s = end_date.date + '-' + end_date.time

            # plot temperature forecasts
            fig, ax = plt.subplots()

            forecasts = np.concatenate((nan_array, dataX[i, -1:, radius, radius, 0], predict[i:i+1, 0]))
            ground_truth = np.concatenate((nan_array, dataX[i, -1:, radius, radius, 0], dataY[i:i+1, 0]))
            network_input = np.concatenate((dataX[i, :, radius, radius, 0], nan_array2))
         
            ax.plot(ind, network_input, 'b-x', label='Network input')
            ax.plot(ind, forecasts, 'r-x', label='Model forecast')
            ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')
            
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            plt.title('Many to One Forecast (' + start_date_s + ' -- ' + end_date_s + ')')
            plt.legend(loc='best')
            plt.savefig(subfolder + '/plot_' + str(i) + '.png')
            plt.cla()

def plot_comparision():
    """
        plot our results and compare them
    """
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    year = 2000
    
    ind = np.arange(1)
    
    fig, ax = plt.subplots()
    for month in months:

        temp_score = np.load('test_conv_lstm/' + 'temperature' + '/month_' + str(month) + '/score.npy')
        constant_score = np.load('test_conv_lstm/' + 'constant_baseline' + '/month_' + str(month) + '/score.npy')

        # temp comparision 
        ax.errorbar(ind, temp_score[0], yerr=temp_score[1], fmt='-o', label='Temperature')
        ax.errorbar(ind, constant_score[0], yerr=constant_score[1], fmt='-o', label='Constant baseline')
       
        plt.xlabel('Forecast Steps')
        plt.ylabel('RMSE (Kelvin)')
        plt.title('Compare temperature forecast models (' + str(month) + '-' + str(year) + ')')
        plt.legend(loc='best')
        plt.savefig("test_conv_lstm/plots/plot_temperature" + str(month) + ".png")
        
        plt.cla()
 
def main():
    test_model()
    evaluate_model()
    evaluate_constant_baseline()
    plot_comparision()
    return 1

if __name__ == "__main__":
    sys.exit(main())
