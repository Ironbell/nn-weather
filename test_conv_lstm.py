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
from network import *

EPOCHS = 20
#GRIB_FOLDER = '/home/isa/sftp/'
GRIB_FOLDER = '/media/isa/VIS1/'
RADIUS = 2

def get_default_data_params():
    params = AttrDict()
    params.steps_before = 20
    params.steps_after = 1
    params.grib_folder = GRIB_FOLDER
    params.forecast_distance = 0
    params.steps_after = 1
    params.lat = 47.25
    params.lon = 8.25#9 + 180
    params.radius = RADIUS 
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
    # train and save the model files
    print ('training started...')
    model_folder = 'test_conv_lstm/' + subfolder_name + '/model'
    trainData = DatasetSquareArea(train_params)
        
    # create and fit the LSTM network
    print('creating model...')
    model_params = get_default_model_params()
    model = create_model(model_params, train_params)
    train_model(model, trainData, EPOCHS, model_folder)
    
def evaluate_constant_baseline(grib_parameters, subfolder_name):
    ''' 
        evaluation of the constant baseline
    '''
    train_params = get_default_data_params()

    # evaluate on whole 2000 and save results
    train_params.years = [2000]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]

    for month in months:
        train_params.months = [month]
        testData = DatasetSquareArea(train_params)
        
        subfolder = 'test_conv_lstm/' + subfolder_name + '/month_' + str(month)
        
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        
        predict = testData.predict_constant(flatten=False)
        dataX, dataY = testData.inverse_transform_data(flatten=False)
        np.save(subfolder + '/prediction.npy', predict)
        
        score = evaluate_model_score_compare(dataY, predict) 
        np.save(subfolder + '/score.npy', score)
        
        # save the error
        np.save(subfolder + '/error.npy', np.abs(dataY[:, :] - predict[:, :]))

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

            forecasts = np.concatenate((nan_array, dataX[i, -1:, RADIUS, RADIUS, 0], predict[i:i+1, 0]))
            ground_truth = np.concatenate((nan_array, dataX[i, -1:, RADIUS, RADIUS, 0], dataY[i:i+1, 0]))
            network_input = np.concatenate((dataX[i, :, RADIUS, RADIUS, 0], nan_array2))
         
            ax.plot(ind, network_input, 'b-x', label='Network input')
            ax.plot(ind, forecasts, 'r-x', label='Constant forecast')
            ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')
            
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            plt.title('Many to One Forecast (' + start_date_s + ' -- ' + end_date_s + ')')
            plt.legend(loc='best')
            plt.savefig(subfolder + '/plot_' + str(i) + '.png')
            plt.cla()
            
            plt.close('all')
            
def evaluate_model(grib_parameters, subfolder_name):
    ''' 
        evaluation of the model
    '''
    train_params = get_default_data_params()

    # evaluate on whole 2000 and save results
    train_params.years = [2000]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    model = load_model('test_conv_lstm/' + subfolder_name + '/model/model.h5')

    for month in months:
        train_params.months = [month]
        testData = DatasetSquareArea(train_params)
        
        subfolder = 'test_conv_lstm/' + subfolder_name + '/month_' + str(month)
        
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        
        # save the score
        predict = testData.predict_data(model, flatten=False)
        dataX, dataY = testData.inverse_transform_data(flatten=False)
        np.save(subfolder + '/prediction.npy', predict)
        
        score = evaluate_model_score_compare(dataY, predict) 
        np.save(subfolder + '/score.npy', score)

        # save the error
        np.save(subfolder + '/error.npy', np.abs(dataY[:, :] - predict[:, :]))

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

            forecasts = np.concatenate((nan_array, dataX[i, -1:, RADIUS, RADIUS, 0], predict[i:i+1, 0]))
            ground_truth = np.concatenate((nan_array, dataX[i, -1:, RADIUS, RADIUS, 0], dataY[i:i+1, 0]))
            network_input = np.concatenate((dataX[i, :, RADIUS, RADIUS, 0], nan_array2))
         
            ax.plot(ind, network_input, 'b-x', label='Network input')
            ax.plot(ind, forecasts, 'r-x', label='Model forecast')
            ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')
            
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            plt.title('Many to One Forecast (' + start_date_s + ' -- ' + end_date_s + ')')
            plt.legend(loc='best')
            plt.savefig(subfolder + '/plot_' + str(i) + '.png')
            plt.cla()
        
        plt.close('all')

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
        temp_press_score = np.load('test_conv_lstm/' + 'temperature_pressure' + '/month_' + str(month) + '/score.npy')
        temp_press_wind_score = np.load('test_conv_lstm/' + 'temperature_pressure_wind' + '/month_' + str(month) + '/score.npy')
        temp_press_dew_score = np.load('test_conv_lstm/' + 'temperature_pressure_dew' + '/month_' + str(month) + '/score.npy')
        constant_score = np.load('test_conv_lstm/' + 'constant_baseline' + '/month_' + str(month) + '/score.npy')

        # temp comparision 
        ax.errorbar(ind, constant_score[0,2], yerr=constant_score[0,3], fmt='-o', label='Constant baseline')
        ax.errorbar(ind, temp_score[0,2], yerr=temp_score[0,3], fmt='-o', label='Temperature')
        ax.errorbar(ind, temp_press_score[0,2], yerr=temp_press_score[0,3], fmt='-o', label='Temperature with Pressure')
        ax.errorbar(ind, temp_press_wind_score[0,2], yerr=temp_press_wind_score[0,3], fmt='-o', label='Temperature with Pressure and Winds')
        ax.errorbar(ind, temp_press_dew_score[0,2], yerr=temp_press_dew_score[0,3], fmt='-o', label='Temperature with Pressure and Dewpoint')

        plt.xlabel('Forecast Steps')
        plt.ylabel('RMSE (Kelvin)')
        plt.title('Compare temperature forecast models (' + str(month) + '-' + str(year) + ')')
        plt.legend(loc='best')
        plt.savefig("test_conv_lstm/plots/score_plot" + str(month) + ".png")
        
        plt.cla()
        
        # boxplots
        temp_error = np.load('test_conv_lstm/' + 'temperature' + '/month_' + str(month) + '/error.npy')
        temp_press_error = np.load('test_conv_lstm/' + 'temperature_pressure' + '/month_' + str(month) + '/error.npy')
        temp_press_wind_error = np.load('test_conv_lstm/' + 'temperature_pressure_wind' + '/month_' + str(month) + '/error.npy')
        temp_press_dew_error = np.load('test_conv_lstm/' + 'temperature_pressure_dew' + '/month_' + str(month) + '/error.npy')
        constant_error = np.load('test_conv_lstm/' + 'constant_baseline' + '/month_' + str(month) + '/error.npy')
        
        data = [constant_error[:,0], temp_error[:,0], temp_press_error[:,0], temp_press_wind_error[:,0], temp_press_dew_error[:,0]]
        plt.boxplot(data)
        plt.xticks([1, 2, 3, 4, 5], ['Constant', 'Temp', 'Temp/Press', 'Temp/Press/Wind', 'Temp/Press/Dew'])
        plt.title('Compare temperature forecast models (' + str(month) + '-' + str(year) + ')')
        plt.ylabel('Error (Kelvin)')
        plt.savefig("test_conv_lstm/plots/boxplot_" + str(month) + ".png")
        
        plt.cla()
        
    plt.close('all')
    
def compare_2():
    """
        plot our results and compare them
    """
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    year = 2000
    
    ind = np.arange(1)
    
    fig, ax = plt.subplots()
    for month in months:

        # boxplots
        temp_press_error = np.load('test_conv_lstm/' + 'temperature_pressure' + '/month_' + str(month) + '/error.npy')
        temp_press_error_16 = np.load('test_conv_lstm/' + 'better' + '/month_' + str(month) + '/error.npy')
        data = [temp_press_error[:,0], temp_press_error_16[:,0]]
        plt.boxplot(data)
        plt.xticks([1, 2], ['Temp/Press 8', 'Better?'])
        plt.title('Compare temperature forecast models (' + str(month) + '-' + str(year) + ')')
        plt.ylabel('Error (Kelvin)')
        plt.savefig("test_conv_lstm/plots/boxplot_comp_" + str(month) + ".png")
        
        plt.cla()
        
    plt.close('all')
 
def main():
    #test_model(['temperature'], 'temperature')
    #evaluate_model(['temperature'], 'temperature')
    #test_model(['temperature', 'pressure'], 'temperature_pressure')
    #evaluate_model(['temperature', 'pressure'], 'temperature_pressure')
    #test_model(['temperature', 'pressure', 'wind_u', 'wind_v'], 'temperature_pressure_wind')
    #evaluate_model(['temperature', 'pressure', 'wind_u', 'wind_v'], 'temperature_pressure_wind')
    #test_model(['temperature', 'pressure', 'dewpoint_temperature'], 'temperature_pressure_dew')
    #evaluate_model(['temperature', 'pressure', 'dewpoint_temperature'], 'temperature_pressure_dew')
    #evaluate_constant_baseline(['temperature'], 'constant_baseline')
    
    
    compare_2() 
    return 1

if __name__ == "__main__":
    sys.exit(main())
