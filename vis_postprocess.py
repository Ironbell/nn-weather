import os
from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from dataset import *
from network import *
from visualise import *

import time

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
    params.lat = 48.75
    params.lon = 2.25
    params.radius = RADIUS
    params.location = 'paris'
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
        if (parameter == 'cloud_cover'):
            string = string + 'Cloud'

    return string

def get_folder_name(lstm_neurons=64, train_months='All', grib_parameters=['temperature'], forecast_distance=0, steps_before=20, lstm_layers=2):
    return \
        'n' + str(lstm_neurons) + '_' + \
        'p' + get_parameter_short(grib_parameters) + '_'  + \
        'm' + train_months.title() + '_' + \
        'f' + str(forecast_distance) + '_' + \
        's' + str(steps_before) + '_' + \
        'l' + str(lstm_layers)

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

def save_flat_error(lstm_neurons=64, train_months='All', grib_parameters=['temperature'], forecast_distance=0, steps_before=20, lstm_layers=2):
    ''' 
        saves the error flat, no abs
    '''
    folder_name = get_folder_name(lstm_neurons, train_months, grib_parameters, forecast_distance, steps_before, lstm_layers)
    subfolder = GRIB_FOLDER + 'data/paris/' + folder_name
    if not os.path.exists(subfolder):
        print('does not exist: ' + subfolder)
        return

    train_params = get_default_data_params()

    train_params.months = get_train_months(train_months)
    train_params.grib_parameters = grib_parameters
    train_params.forecast_distance = forecast_distance
    train_params.steps_before = steps_before

    # now evaluate
    print ('loading data for testing...')
    train_params.years = TEST_YEARS
    train_params.months = list(range(1, 12 + 1))
    testData = DatasetSquareArea(train_params)

    predict = np.load(subfolder + '/predict.npy')
    dataX = np.load(subfolder + '/dataX.npy')
    dataY = np.load(subfolder + '/dataY.npy')

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
                if (len(month_data) % 4 > 0):
                    month_data = ([np.zeros((dataY.shape[1]))] * (4 - (len(month_data) % 4))) + month_data
                np.save(y_subfolder + '/' + str(current_month) + '_error_flat.npy', np.asarray(month_data).reshape((-1, 4, channels)))
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
            
        error = predict[step, :] - dataY[step, :]
        month_data.append(error)
        
    if (len(month_data) > 0):
        if (len(month_data) % 4 > 0):
            month_data = ([np.zeros((dataY.shape[1]))] * (4 - (len(month_data) % 4))) + month_data       
        np.save(y_subfolder + '/' + str(current_month) + '_error_flat.npy', np.asarray(month_data).reshape((-1, 4, channels)))
        year_overview[year_it, month_it, :] = np.mean(np.asarray(month_data), axis=0)

    np.save(subfolder + '/year_overview_flat.npy', year_overview)
    np.save(subfolder + '/score_flat.npy', np.mean(year_overview.reshape((-1, channels)), axis=0))
    
def save_abs_error(lstm_neurons=64, train_months='All', grib_parameters=['temperature'], forecast_distance=0, steps_before=20, lstm_layers=2):
    ''' 
        saves the abs error
    '''
    folder_name = get_folder_name(lstm_neurons, train_months, grib_parameters, forecast_distance, steps_before, lstm_layers)
    subfolder = GRIB_FOLDER + 'data/paris/' + folder_name
    if not os.path.exists(subfolder):
        print('does not exist: ' + subfolder)
        return

    train_params = get_default_data_params()

    train_params.months = get_train_months(train_months)
    train_params.grib_parameters = grib_parameters
    train_params.forecast_distance = forecast_distance
    train_params.steps_before = steps_before

    # now evaluate
    print ('loading data for testing...')
    train_params.years = TEST_YEARS
    train_params.months = list(range(1, 12 + 1))
    testData = DatasetSquareArea(train_params)

    predict = np.load(subfolder + '/predict.npy')
    dataX = np.load(subfolder + '/dataX.npy')
    dataY = np.load(subfolder + '/dataY.npy')

    # pre-save year overview and month
    years = train_params.years
    months = train_params.months

    current_year = ''
    current_month = ''
    year_it = -1
    month_it = -1
    month_data = []
    std_data = []
    channels = len(train_params.grib_parameters)
    
    year_overview = np.empty((len(years), len(months), channels))
    year_overview_std = np.empty((len(years), len(months), channels))
    
    for step in range(dataY.shape[0]):
        frame_data = testData.frames_data[testData.frames_idx[step]]
        
        if (current_month != frame_data.month()):
            if (len(month_data) > 0):
                if (len(month_data) % 4 > 0):
                    month_data = ([np.zeros((dataY.shape[1]))] * (4 - (len(month_data) % 4))) + month_data
                np.save(y_subfolder + '/' + str(current_month) + '_error.npy', np.asarray(month_data).reshape((-1, 4, channels)))
                year_overview[year_it, month_it, :] = np.mean(np.asarray(month_data), axis=0)
                year_overview_std[year_it, month_it, :] = np.std(np.asarray(std_data), axis=0)
                month_data = []
                std_data = []
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
            
        error = np.abs(predict[step, :] - dataY[step, :])
        month_data.append(error)
        std_data.append(dataY[step, :])
        
    if (len(month_data) > 0):
        if (len(month_data) % 4 > 0):
            month_data = ([np.zeros((dataY.shape[1]))] * (4 - (len(month_data) % 4))) + month_data       
        np.save(y_subfolder + '/' + str(current_month) + '_error.npy', np.asarray(month_data).reshape((-1, 4, channels)))
        year_overview[year_it, month_it, :] = np.mean(np.asarray(month_data), axis=0)
        year_overview_std[year_it, month_it, :] = np.std(np.asarray(std_data), axis=0)
        month_data = []
        std_data = []

    np.save(subfolder + '/year_overview.npy', year_overview)
    np.save(subfolder + '/year_overview_std.npy', year_overview_std)
    np.save(subfolder + '/score_std.npy', np.std(dataY, axis=0))
    np.save(subfolder + '/score.npy', np.mean(year_overview.reshape((-1, channels)), axis=0))
    
def save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature'], forecast_distance=0, steps_before=20, lstm_layers=2):
    save_abs_error(lstm_neurons, train_months, grib_parameters, forecast_distance, steps_before, lstm_layers)
    save_flat_error(lstm_neurons, train_months, grib_parameters, forecast_distance, steps_before, lstm_layers)

def main(): 
    save_errors(lstm_neurons=64, train_months='Summer', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=0, steps_before=20)
    
    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=0, steps_before=20)
    
    save_errors(lstm_neurons=64, train_months='Winter', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=0, steps_before=20)
    
    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'cloud_cover'], forecast_distance=0, steps_before=20)
    
    save_errors(lstm_neurons=128, train_months='All', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=0, steps_before=20)
    
    save_errors(lstm_neurons=32, train_months='All', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=0, steps_before=20)
    
    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=1, steps_before=20)
    
    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=4, steps_before=20)
    
    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=0, steps_before=30)

    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=0, steps_before=10)
    
    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'wind_u', 'wind_v'], forecast_distance=0, steps_before=20)
    
    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'u_ambient', 'v_ambient'], forecast_distance=0, steps_before=20)
    
    #save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'u_features', 'v_features'], forecast_distance=0, steps_before=20)
    
    #save_errors(lstm_neurons=128, train_months='All', grib_parameters=['temperature', 'u_features', 'v_features', 'u_ambient', 'v_ambient'], forecast_distance=0, steps_before=20)
    
    save_errors(lstm_neurons=128, train_months='All', grib_parameters=['temperature', 'surface_pressure', 'cloud_cover'], forecast_distance=0, steps_before=20)
    
    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature'], forecast_distance=0, steps_before=20)
    
    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'surface_pressure'], forecast_distance=0, steps_before=20, lstm_layers=4) 
    
    save_errors(lstm_neurons=64, train_months='All', grib_parameters=['temperature', 'surface_pressure', 'cloud_cover'], forecast_distance=0, steps_before=20)

    return 1

if __name__ == "__main__":
    sys.exit(main())
