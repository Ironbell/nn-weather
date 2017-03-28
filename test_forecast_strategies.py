from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from network import *
from visualise import *

EPOCHS = 20

def test_many_to_many_forecast():
    ''' 
        testing how well the network can predict
        in a many to many scenario 
    '''
    train_params = AttrDict()
    train_params.steps_before = 30
    train_params.forecast_distance = 0
    train_params.steps_after = 8
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.npoints = 25
    train_params.grib_folder = '/home/isa/sftp/temperature/'
    train_params.months = [1,2,3,4,5,6,7,8,9,10,11,12]
    train_params.years = [2000,2001,2002]

    #train and save the model files
    print ('training started...')
    model_folder = 'test_forecast_strategies/model_mtm'
    trainData = DatasetNearest(train_params)
        
    # create and fit the LSTM network
    print('creating model...')
    model = create_model(train_params.steps_before, train_params.steps_after, trainData.nb_features, train_params.npoints * 2)
    train_model(model, trainData, EPOCHS, model_folder)

    # evaluate on whole 2003 and save results
    train_params.years = [2003]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]

    for month in months:
        train_params.months = [month]
        testData = DatasetNearest(train_params)
        predict = testData.predict_data(model)
        np.save('test_forecast_strategies/mtm_prediction_' + str(month) + '.npy', predict)
        score = evaluate_model_score(model, testData)
        np.save('test_forecast_strategies/mtm_score_' + str(month) + '.npy', score)
        
def test_recursive_forecast():
    ''' 
        testing how well the network does
        using a recursive model for the forecast
    '''
    recursive_steps = 8
    train_params = AttrDict()
    train_params.steps_before = 30
    train_params.forecast_distance = 0
    train_params.steps_after = 1
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.npoints = 25
    train_params.grib_folder = '/home/isa/sftp/temperature/'
    train_params.months = [1,2,3,4,5,6,7,8,9,10,11,12]
    train_params.years = [2000,2001,2002]

    #train and save the model files
    print ('training started...')
    model_folder = 'test_forecast_strategies/model_rec'
    trainData = DatasetNearest(train_params)
        
    # create and fit the LSTM network
    print('creating model...')
    model = create_model(train_params.steps_before, train_params.steps_after, trainData.nb_features, train_params.npoints * 2)
    train_model(model, trainData, EPOCHS, model_folder)

    # evaluate on whole 2003 and save results
    train_params.years = [2003]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]

    for month in months:
        train_params.months = [month]
        train_params.steps_after = 1
        testData = DatasetNearest(train_params)
        train_params.steps_after = recursive_steps
        testData2 = DatasetNearest(train_params)
        predict = predict_multiple(model, testData, recursive_steps)
        for i in range(predict.shape[0]):
            predict[i] = testData.scaler.inverse_transform(predict[i])
        np.save('test_forecast_strategies/rec_prediction_' + str(month) + '.npy', predict)
        _, dataY = testData2.inverse_transform_data()
        score = evaluate_model_score_raw(dataY, predict[0:dataY.shape[0]])
        np.save('test_forecast_strategies/rec_score_' + str(month) + '.npy', score)
        
def test_individual_model_forecast():
    ''' 
        testing how well the network can predict
        with individual models
    '''
    train_params = AttrDict()
    train_params.steps_before = 30
    train_params.steps_after = 1
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.npoints = 25
    train_params.grib_folder = '/home/isa/sftp/temperature/'
    train_params.months = [1,2,3,4,5,6,7,8,9,10,11,12]
    train_params.years = [2000,2001,2002]
    
    forecast_distance_list = [1,2,3,4,5,6,7,8]
    
    # train the individual models
    for forecast_distance in forecast_distance_list:
        model_folder = 'test_forecast_strategies/model_im_' + str(forecast_distance)
        train_params.forecast_distance = forecast_distance
        trainData = DatasetNearest(train_params)
        
        print('creating model...')
        model = create_model(train_params.steps_before, train_params.steps_after, trainData.nb_features, train_params.npoints * 2)
        train_model(model, trainData, EPOCHS, model_folder)
       
    # evaluate the individual models
    # evaluate on whole 2003 and save results
    train_params.years = [2003]
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    true_steps = forecast_distance_list[-1]

    for month in months:
        train_params.months = [month]
        train_params.steps_after = 1
        testData = DatasetNearest(train_params)
        train_params.steps_after = true_steps
        testData2 = DatasetNearest(train_params)
        
        predict = np.empty((testData.dataY.shape[0], testData2.dataY.shape[1], testData.dataY.shape[2]))
        for forecast_distance in forecast_distance_list:
            model_folder = 'test_forecast_strategies/model_im_' + str(forecast_distance)
            model = load_model(model_folder + '/model.h5')
            predict[:, forecast_distance-1:forecast_distance, :] = testData.predict_data(model)
       
        np.save('test_forecast_strategies/im_prediction_' + str(month) + '.npy', predict)
        _, dataY = testData2.inverse_transform_data()
        score = evaluate_model_score_raw(dataY, predict[0:dataY.shape[0]])
        np.save('test_forecast_strategies/im_score_' + str(month) + '.npy', score)
        
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

def test_forecast_strategies():
    ''' 
        compare three strategies for longer step size
        1. many to many LSTM
        2. individual models 
        3. recursive model
    '''
    #test_many_to_many_forecast()
    #test_recursive_forecast()
    #test_individual_model_forecast()
    plot_forecast_strategies()
    '''# plot
    nan_array = np.empty((testData.params.steps_before - 1))
    nan_array.fill(np.nan)
    nan_array2 = np.empty(train_params.steps_after)
    nan_array2.fill(np.nan)
    ind = np.arange(testData.params.steps_before + train_params.steps_after)

    for i in range(10):
        plt.cla()
        fig, ax = plt.subplots()
        forecasts = np.concatenate((nan_array, dataX[i, -1:, 0], predict[i, :, 0]))
        ground_truth = np.concatenate((nan_array, dataX[i, -1:, 0], dataY[i, :, 0]))
        network_input = np.concatenate((dataX[i, :, 0], nan_array2))
     
        ax.plot(ind, network_input, 'b-x', label='Network input')
        ax.plot(ind, forecasts, 'r-x', label='Many to many model forecast')
        ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')

        start_date = testData.frames_data[testData.frames_idx[i + testData.params.steps_before]]
        end_date = testData.frames_data[testData.frames_idx[i + testData.params.steps_before + testData.params.steps_after - 1]]
        
        start_date_s = start_date.date + '-' + start_date.time
        end_date_s = end_date.date + '-' + end_date.time
        
        plt.xlabel('Time (6h steps)')
        plt.ylabel('Temperature (Kelvin)')
        plt.title('Many to Many Forecast (' + start_date_s + ' -- ' + end_date_s + ')')
        plt.legend(loc='upper right')
        plt.savefig("test_many_to_many/plot" + str(i) + ".png")'''

def main():
    test_forecast_strategies()
    return 1

if __name__ == "__main__":
    sys.exit(main())
