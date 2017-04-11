import os, json
from attrdict import AttrDict
from keras.models import load_model
from keras.utils import plot_model
import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, TimeDistributed, Flatten, RepeatVector
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.metrics import mean_squared_error

from dataset import *
from visualise import *

EPOCHS = 100
#GRIB_FOLDER = '/home/isa/sftp/'
GRIB_FOLDER = '/media/isa/VIS1/'
STEPS_BEFORE = 20
RADIUS = 2

def get_default_data_params():
    params = AttrDict()
    params.steps_before = 20
    params.steps_after = 1
    params.grib_folder = GRIB_FOLDER
    params.forecast_distance = 0
    params.steps_after = 1
    params.lat = 47.25
    params.lon = 8.25
    params.radius = 2
    params.grib_parameters = ['temperature', 'pressure']
    params.months = [6, 7, 8]#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    params.years = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]
    params.hours = [0, 6, 12, 18]
    return params

def create_model(steps_before, radius, nb_features):
    """ 
        creates, compiles and returns a RNN model 
        @param steps_before: the number of previous time steps (input). 
        @param radius: the radius of the square around the feature of interest
    """
    DROPOUT = 0.2
    HIDDEN_NEURONS = 64
    diameter = 1 + 2 * radius

    model = Sequential()
    model.add(TimeDistributed(Conv2D(filters=8, kernel_size=(2,2), padding='same', input_shape=(diameter, diameter, nb_features)), input_shape=(steps_before, diameter, diameter, nb_features)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(HIDDEN_NEURONS)))
    model.add(LSTM(HIDDEN_NEURONS, return_sequences=False))
    model.add(Dropout(DROPOUT))
    model.add(RepeatVector(1))
    model.add(LSTM(HIDDEN_NEURONS, return_sequences=True))
    model.add(TimeDistributed(Dense(nb_features)))
    model.add(TimeDistributed(Activation('linear')))  

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

def test_model(grib_parameters, subfolder_name):
    ''' 
        train the network
    '''
    train_params = get_default_data_params()

    # train and save the model files
    print ('training started...')
    model_folder = 'test_tp_seasons/' + subfolder_name + '/model'
    trainData = DatasetSquareArea(train_params)
        
    # create and fit the LSTM network
    print('creating model...')
    model = create_model(train_params.steps_before, train_params.radius, len(train_params.grib_parameters))
    train_model(model, trainData, EPOCHS, model_folder)
                
def evaluate_model(grib_parameters, subfolder_name):
    ''' 
        evaluation of the model
    '''
    train_params = get_default_data_params()

    # evaluate on whole 2000 and save results
    years = list(range(1990, 2017))
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    model = load_model('test_tp_seasons/' + subfolder_name + '/model/model.h5')
    plot_model(model, to_file='test_tp_seasons/' + subfolder_name + '/model/model.png')

    score = np.empty((len(years), len(months)))
    year_it = 0;
    for year in years:
        month_it = 0;
        for month in months:
            train_params.months = [month]
            train_params.years = [year]
            testData = DatasetSquareArea(train_params)
            
            # save the score
            predict = testData.predict_data(model, flatten=False)
            dataX, dataY = testData.inverse_transform_data(flatten=False)

            avg_score = evaluate_model_score_compare(dataY, predict) 
            np.save('test_tp_seasons/' + subfolder_name + '/' + '/avg_score_' + str(year) + '_' + str(month) + '.npy', avg_score)
            score[year_it, month_it] = avg_score[0, 0, 4] #RMSE
            
            np.save('test_tp_seasons/' + subfolder_name + '/' + '/score.npy', score)

            month_it = month_it + 1
        year_it = year_it + 1

def plot_score(subfolder_name):
    score = np.load('test_tp_seasons/' + subfolder_name + '/score.npy')
    display_score(score, [1990,2016,1,12], 'test_tp_seasons/' + subfolder_name + '_comparision.png')

def main():
    #test_model(['temperature', 'pressure'], 'summer')
    evaluate_model(['temperature', 'pressure'], 'summer')
    plot_score('summer')
    return 1

if __name__ == "__main__":
    sys.exit(main())
