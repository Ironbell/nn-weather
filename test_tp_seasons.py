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
from network import *

EPOCHS = 100
#GRIB_FOLDER = '/home/isa/sftp/'
GRIB_FOLDER = '/media/isa/VIS1/'
STEPS_BEFORE = 20
RADIUS = 3

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
    params.is_zurich = True
    params.grib_parameters = ['temperature', 'pressure']
    params.months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    params.years = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]
    params.hours = [0, 6, 12, 18]
    return params

def test_model(grib_parameters, subfolder_name):
    ''' 
        train the network
    '''
    train_params = get_default_data_params()
    model_params = get_default_model_params()

    # train and save the model files
    print ('training started...')
    model_folder = 'test_tp_seasons/' + subfolder_name + '/model'
    trainData = DatasetSquareArea(train_params)
        
    # create and fit the LSTM network
    print('creating model...')
    model = create_model(model_params, train_params)
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
            score[year_it, month_it] = avg_score[0, 4] #RMSE
            
            np.save('test_tp_seasons/' + subfolder_name + '/' + '/score.npy', score)

            month_it = month_it + 1
        year_it = year_it + 1

def plot_score(subfolder_name):
    score = np.load('test_tp_seasons/' + subfolder_name + '/score.npy')
    display_score(score, [1990,2016,12,1], 'test_tp_seasons/' + subfolder_name + '_comparision.png')

def main():
    test_model(['temperature', 'pressure'], 'all')
    evaluate_model(['temperature', 'pressure'], 'all')
    plot_score('all')
    return 1

if __name__ == "__main__":
    sys.exit(main())
