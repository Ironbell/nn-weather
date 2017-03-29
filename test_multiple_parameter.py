import os
from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from network import *
from visualise import *

EPOCHS = 100

def test_features(grib_parameters, subfolder_name):
    ''' 
        training a network with one or more grib features
    '''
    train_params = AttrDict()
    train_params.steps_before = 30
    train_params.forecast_distance = 0
    train_params.steps_after = 8
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.npoints = 9
    train_params.grib_folder = '/media/isa/VIS1/'
    train_params.grib_parameters = grib_parameters
    train_params.months = [1,2,3,4,5,6,7,8,9,10,11,12]
    train_params.years = [2000,2001,2002,2003]

    # train and save the model files
    print ('training started...')
    model_folder = 'test_multiple_parameters/' + subfolder_name + '/model'
    trainData = DatasetNearest(train_params)
        
    # create and fit the LSTM network
    print('creating model...')
    model = create_model(train_params.steps_before, train_params.steps_after, trainData.params.nb_features, trainData.params.nb_features * 2)
    train_model(model, trainData, EPOCHS, model_folder)

def evaluate_features(grib_parameters, subfolder_name):
    ''' 
        evaluation of how well the network can predict
    '''
    train_params = AttrDict()
    train_params.steps_before = 30
    train_params.forecast_distance = 0
    train_params.steps_after = 8
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.npoints = 9
    train_params.grib_folder = '/media/isa/VIS1/'
    train_params.grib_parameters = grib_parameters
    train_params.years = [2004]

    model = load_model('test_multiple_parameters/' + subfolder_name + '/model/model.h5')

    # evaluate on whole 2003 and save results
    train_params.years = [2004]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    for month in months:
        train_params.months = [month]
        testData = DatasetNearest(train_params)
        
        subfolder = 'test_multiple_parameters/' + subfolder_name + '/month_' + str(month)
        
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        
        predict = testData.predict_data(model, flatten=False)
        dataX, dataY = testData.inverse_transform_data(flatten=False)
        np.save(subfolder + '/prediction.npy', predict)
        
        score = evaluate_model_score(model, testData)
        score = score[:,0:len(train_params.grib_parameters),:] # we're only interested in the first feature
        np.save(subfolder + '/score.npy', score)
        print('score shape:')
        print(score.shape)

        # plot the first results as a sanity check
        nb_results = 10
        
        nan_array = np.empty((testData.params.steps_before - 1))
        nan_array.fill(np.nan)
        nan_array2 = np.empty(train_params.steps_after)
        nan_array2.fill(np.nan)
        ind = np.arange(testData.params.steps_before + train_params.steps_after)

        for i in range(nb_results):
            start_date = testData.frames_data[testData.frames_idx[i + testData.params.steps_before]]
            end_date = testData.frames_data[testData.frames_idx[i + testData.params.steps_before + testData.params.steps_after - 1]]
            
            start_date_s = start_date.date + '-' + start_date.time
            end_date_s = end_date.date + '-' + end_date.time

            # plot temperature forecasts
            fig, ax = plt.subplots()
            for j in range(len(testData.params.grib_parameters)):
                forecasts = np.concatenate((nan_array, dataX[i, -1:, 0, j], predict[i, :, 0, j]))
                ground_truth = np.concatenate((nan_array, dataX[i, -1:, 0, j], dataY[i, :, 0, j]))
                network_input = np.concatenate((dataX[i, :, 0, j], nan_array2))
             
                ax.plot(ind, network_input, 'b-x', label='Network input')
                ax.plot(ind, forecasts, 'r-x', label='Many to many model forecast')
                ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')
                
                plt.xlabel('Time (6h steps)')
                plt.ylabel(testData.params.grib_parameters[j])
                plt.title('Many to Many Forecast (' + start_date_s + ' -- ' + end_date_s + ')')
                plt.legend(loc='best')
                plt.savefig(subfolder + '/plot_' + testData.params.grib_parameters[j] + str(i) + '.png')
                plt.cla()
                
def plot_comparision():
    """
        plot our three results and compare them
    """
    runs = ['temperature_only', 'pressure_only', 'temperature_pressure']
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    year = 2004
    
    ind = np.arange(8)
    
    fig, ax = plt.subplots()
    for month in months:

        temp_score = np.load('test_multiple_parameters/' + 'temperature_only' + '/month_' + str(month) + '/score.npy')
        press_score = np.load('test_multiple_parameters/' + 'pressure_only' + '/month_' + str(month) + '/score.npy')
        temp_press_score = np.load('test_multiple_parameters/' + 'temperature_pressure' + '/month_' + str(month) + '/score.npy')
      
      
        # temp comparision 
        ax.errorbar(ind, temp_score[:, 0, 0], yerr=temp_score[:, 0, 1], fmt='-o', label='Only temperature')
        ax.errorbar(ind, temp_press_score[:, 0, 0], yerr=temp_press_score[:, 0, 1], fmt='-o', label='Temperature combined with pressure')
       
        plt.xlabel('Forecast Steps')
        plt.ylabel('RMSE (Kelvin)')
        plt.title('Compare temperature forecast models (' + str(month) + '-' + str(year) + ')')
        plt.legend(loc='best')
        plt.savefig("test_multiple_parameters/plots/plot_temperature" + str(month) + ".png")
        
        plt.cla()
        
        # pressure comparision 
        ax.errorbar(ind, press_score[:, 0, 0], yerr=press_score[:, 0, 1], fmt='-o', label='Only pressure')
        ax.errorbar(ind, temp_press_score[:, 1, 0], yerr=temp_press_score[:, 1, 1], fmt='-o', label='Pressure combined with temperature')
       
        plt.xlabel('Forecast Steps')
        plt.ylabel('Pressure (Pascal)')
        plt.title('Compare pressure forecast models (' + str(month) + '-' + str(year) + ')')
        plt.legend(loc='best')
        plt.savefig("test_multiple_parameters/plots/plot_pressure" + str(month) + ".png")
        
        plt.cla()
    
def main():
    #test_features(['temperature'], 'temperature_only')
    evaluate_features(['temperature'], 'temperature_only')
    test_features(['pressure'], 'pressure_only')
    evaluate_features(['pressure'], 'pressure_only')
    test_features(['temperature', 'pressure'], 'temperature_pressure')
    evaluate_features(['temperature', 'pressure'], 'temperature_pressure')
    plot_comparision()
    return 1

if __name__ == "__main__":
    sys.exit(main())
