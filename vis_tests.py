import os
from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from dataset import *
from network import *
from visualise import *

EPOCHS = 100
GRIB_FOLDER = '/media/isa/VIS1/'
RADIUS = 3
TEST_YEARS = list(range(1990, 2016 + 1))

def get_default_data_params():
    params = AttrDict()
    params.steps_before = 20
    params.steps_after = 1
    params.grib_folder = GRIB_FOLDER
    params.forecast_distance = 0
    params.steps_after = 1
    params.lat = 47.25
    params.lon = 8.25
    params.radius = RADIUS
    params.location = 'zurich'
    params.grib_parameters = ['temperature', 'pressure']
    params.months = list(range(1, 12 + 1))
    params.years = list(range(1990, 2000))
    params.hours = [0, 6, 12, 18]
    return params
    
def get_parameter_short(grib_parameters):
    string = ''
    for parameter in grib_parameters:
        if (parameter == 'temperature'):
            string = string + 'Temp'
        if (parameter == 'pressure'):
            string = string + 'Press'
        if (parameter == 'surface_pressure'):
            string = string + 'Spress'
        if (parameter == 'wind_u'):
            string = string + 'Wu'
        if (parameter == 'wind_v'):
            string = string + 'Wv'
        if (parameter == 'u_ambient'):
            string = string + 'Av'
        if (parameter == 'v_ambient'):
            string = string + 'Au'
        if (parameter == 'u_features'):
            string = string + 'Fu'
        if (parameter == 'v_features'):
            string = string + 'Fv'

    return string

def get_folder_name(lstm_neurons=64, train_months='All', grib_parameters=['temperature'], forecast_distance=0, steps_before=20):
    return \
        'n' + str(lstm_neurons) + '_' + \
        'p' + get_parameter_short(grib_parameters) + '_'  + \
        'm' + train_months.title() + '_' + \
        'f' + str(forecast_distance) + '_' + \
        's' + str(steps_before)

def get_train_months(months):
    if (months == 'Summer'):
        return [6, 7, 8]
    if (months == 'Winter'):
        return [1, 2, 12]
    if (months == 'Fall'):
        return [9, 10, 11]
    if (months == 'Spring'):
        return [3, 4, 5]
    return list(range(1, 13))

def run_test(lstm_neurons=64, train_months='all', grib_parameters=['temperature'], forecast_distance=0, steps_before=20):
    ''' 
        runs a test for the visualisation model.
    '''
    folder_name = get_folder_name(lstm_neurons, train_months, grib_parameters, forecast_distance, steps_before)
    subfolder = GRIB_FOLDER + 'data/' + folder_name
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    train_params = get_default_data_params()
    model_params = get_default_model_params()

    model_params.lstm_neurons = lstm_neurons
    train_params.months = get_train_months(train_months)
    train_params.grib_parameters = grib_parameters
    train_params.forecast_distance = forecast_distance
    train_params.steps_before = steps_before
    
    print ('test started: ' + folder_name)
    
    #train and save the model files
    print ('loading data for training...')
    trainData = DatasetSquareArea(train_params)
    model_folder = subfolder + '/model'

    # create and fit the network
    print('creating model')
    model = create_model(model_params, train_params)
    train_model(model, trainData, EPOCHS, model_folder)
    
    #model = load_model(subfolder + '/model/model.h5')
   
    # now evaluate
    print ('loading data for testing...')
    train_params.years = TEST_YEARS
    train_params.months = list(range(1, 12 + 1))
    testData = DatasetSquareArea(train_params)

    dataX, dataY = testData.inverse_transform_data()
    dataX = dataX[:, :, RADIUS, RADIUS, :]
    predict = testData.predict_data(model)

    np.save(subfolder + '/predict.npy', predict)
    np.save(subfolder + '/dataX.npy', dataX)
    np.save(subfolder + '/dataY.npy', dataY)
    
    # pre-save year overview and month
    years = train_params.years
    months = train_params.months

    current_year = ''
    current_month = ''
    year_it = -1
    month_it = -1
    month_data = []
    channels = len(train_params.grib_parameters)
    
    year_overview = np.empty((len(years), len(months), channels))
    
    for step in range(dataY.shape[0]):
        frame_data = testData.frames_data[testData.frames_idx[step]]
        
        if (current_month != frame_data.month()):
            if (len(month_data) > 0):
                np.save(y_subfolder + '/' + str(current_month) + '_error.npy', np.asarray(month_data).reshape((-1, 4, channels)))
                year_overview[year_it, month_it, :] = np.mean(np.asarray(month_data), axis=0)
                month_data = []
                current_month = frame_data.month()
                month_it = month_it + 1
                
        if (current_year != frame_data.year()):
            current_year = frame_data.year()
            current_month = frame_data.month()
            
            y_subfolder = subfolder + '/' + str(current_year)
            if not os.path.exists(y_subfolder):
                os.makedirs(y_subfolder)   
 
            month_it = 0
            year_it = year_it + 1
            
        error = np.absolute(dataY[step, :] - predict[step, :])
        month_data.append(error)
        
    if (len(month_data) > 0):
        np.save(y_subfolder + '/' + str(current_month) + '_error.npy', np.asarray(month_data).reshape((-1, 4, channels)))
        year_overview[year_it, month_it, :] = np.mean(np.asarray(month_data), axis=0)
        
    np.save(subfolder + '/year_overview.npy', year_overview)
            
    for channel in range(channels):
        display_score(year_overview[:,:,channel], [1990,2016,12,1], subfolder + '/' + str(channel) + '_season_comparision.png')
            

def season_plot(lstm_neurons=64, train_months='all', grib_parameters=['temperature'], forecast_distance=0, steps_before=20):
    folder_name = get_folder_name(lstm_neurons, train_months, grib_parameters, forecast_distance, steps_before)
    subfolder = GRIB_FOLDER + 'data/' + folder_name
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    train_params = get_default_data_params()
    model_params = get_default_model_params()

    model_params.lstm_neurons = lstm_neurons
    train_params.months = get_train_months(train_months)
    train_params.grib_parameters = grib_parameters
    train_params.forecast_distance = forecast_distance
    train_params.steps_before = steps_before
    
    years = list(range(1990, 2017))
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    
    model = load_model(subfolder + '/model/model.h5')
  
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
            score[year_it, month_it] = avg_score[0, 4] #RMSE
            
            np.save(subfolder + '/score.npy', score)

            month_it = month_it + 1
        year_it = year_it + 1
        
    display_score(score, [1990,2016,12,1], subfolder + '/season_comparision.png')

def main():
    run_test(lstm_neurons=32, train_months='Summer', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=0, steps_before=20)
    #season_plot(lstm_neurons=64, train_months='Summer', grib_parameters=['temperature', 'pressure'], forecast_distance=0, steps_before=20)
    return 1

if __name__ == "__main__":
    sys.exit(main())
