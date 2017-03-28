from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from network import *
from visualise import *

EPOCHS = 20

def test_multiple_parameter():
    ''' 
        testing how well the network can predict
        with multiple parameters. 
    '''
    train_params = AttrDict()
    train_params.steps_before = 30
    train_params.forecast_distance = 0
    train_params.steps_after = 8
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.npoints = 9
    train_params.grib_folder = '/media/isa/VIS1/'
    train_params.grib_parameters = ['temperature']
    train_params.months = [1]#[1,2,3,4,5,6,7,8,9,10,11,12]
    train_params.years = [2000]#[2000,2001,2002]

    # train and save the model files
    print ('training started...')
    model_folder = 'test_multiple_parameters/model_mtm'
    trainData = DatasetNearest(train_params)
        
    # create and fit the LSTM network
    print('creating model...')
    model = create_model(train_params.steps_before, train_params.steps_after, trainData.params.nb_features, trainData.params.nb_features * 2)
    train_model(model, trainData, EPOCHS, model_folder)

    # evaluate on whole 2003 and save results
    train_params.years = [2000]
    months = [1]
    testData = DatasetNearest(train_params)
    predict = testData.predict_data(model, flatten=False)
    dataX, dataY = testData.inverse_transform_data(flatten=False)
    np.save('test_multiple_parameters/mtm_prediction.npy', predict)

    # plot as a sanity check
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
            plt.savefig("test_multiple_parameters/plot_" + testData.params.grib_parameters[j] + str(i) + ".png")
            plt.cla()

def plot_forecast_strategies():
    """
        compare the three forecast strategies in one 
        plot per month
    """
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    year = 2003
    
    ind = np.arange(8)
    
    for month in months:
        mtm_score = np.load('test_forecast_strategies/mtm_score_' + str(month) + '.npy')
        im_score = np.load('test_forecast_strategies/im_score_' + str(month) + '.npy')
        rec_score = np.load('test_forecast_strategies/rec_score_' + str(month) + '.npy')
        
        plt.cla()
        fig, ax = plt.subplots()

        ax.errorbar(ind, mtm_score[:, 0, 0], yerr=mtm_score[:, 0, 1], fmt='-o', label='Many to many model forecast')
        ax.errorbar(ind, im_score[:, 0, 0], yerr=im_score[:, 0, 1], fmt='-o', label='Individual model forecast')
        ax.errorbar(ind, rec_score[:, 0, 0], yerr=rec_score[:, 0, 1], fmt='-o', label='Recursive model forecast')

        plt.xlabel('Forecast Steps')
        plt.ylabel('RMSE (Kelvin)')
        plt.title('Compare forecast models (' + str(month) + '-2013)')
        plt.legend(loc='lower right')
        plt.savefig("test_forecast_strategies/plots/plot_" + str(month) + ".png")

def main():
    test_multiple_parameter()
    return 1

if __name__ == "__main__":
    sys.exit(main())
