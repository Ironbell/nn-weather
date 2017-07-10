import os, sys, gc
from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from network import *
from visualise import *
from eccodes import *

import time

GRIB_FOLDER = '/media/isa/VIS1/'
RADIUS = 3
TEST_YEARS = [2016]
CHANNELS4 = ['cloud_cover', 'temperature'] # channels with 4 steps
CHANNELS2 = ['total_precipitation'] # channels with 2 steps

LATLON = [[47.25,8.25],[47.25,4.5],[33.75,241.5],[54.0,26.25],[24.75,357.75],[-1.5,24.75],[-24.0,45.75],[4.5,293.25],[-21.0,308.25],[-48.0,288.75],[-43.5,170.25],[-22.5,125.25],[19.5,78.75],[64.5,60.75],[57.75,107.25],[66.0,168.75],[73.5,316.5],[-79.5,302.25],[60.75,261.75],[30.0,277.5],[31.5,196.5],[-27.0,77.25],[28.5,315.75]]

def get_default_data_params():
    params = AttrDict()
    params.steps_before = 20
    params.steps_after = 1
    params.grib_folder = GRIB_FOLDER
    params.forecast_distance = 0
    params.radius = RADIUS
    params.location = ''
    params.grib_parameters = ['temperature', 'surface_pressure']
    params.months = list(range(1, 12 + 1))
    params.years = TEST_YEARS
    params.hours = [0, 6, 12, 18]
    return params

def calculate_latlon(lat, lon):
    if (lon < 0):
        lon = lon + 360
    lon = round_nearest(lon, 0.75)
    lat = round_nearest(lat, 0.75)
    #print('[' + str(lat) + ',' + str(lon) + ']', end='')
    
def print_latlon():
    print('[')
    calculate_latlon(47.012951, 8.101918)
    calculate_latlon(47.608861, 4.762074)
    calculate_latlon(34.052234, -118.243685)
    calculate_latlon(53.701431, 26.399159)
    calculate_latlon(24.454415, -2.428966)
    calculate_latlon(-1.134523, 24.641347)
    calculate_latlon(-24.279752, 45.383534)
    calculate_latlon(4.485820, -66.413383)
    calculate_latlon(-21.036036, -51.647758)
    calculate_latlon(-48.277877, -70.983696)
    calculate_latlon(-43.638245, 170.188265)
    calculate_latlon(-22.342688, 125.539827)
    calculate_latlon(19.567321, 78.430452)
    calculate_latlon(64.133270, 60.852242)
    calculate_latlon(57.661432, 107.258492)
    calculate_latlon(65.770006, 168.781929)
    calculate_latlon(73.704189, -43.210173)
    calculate_latlon(-79.447016, -57.624235)
    calculate_latlon(60.547701, -98.405571)
    calculate_latlon(30.076003, -82.233696)
    calculate_latlon(31.585337, -163.444633)
    calculate_latlon(-26.816947, 77.024202)
    calculate_latlon(28.234015, -44.264860)
    print(']')

def classify_weather(latitude=47.25, longitude=8.25):
    ''' 
        classifies the weather for a given location, for each step in TEST_YEARS
    '''

    for year in TEST_YEARS:
        files4 = []
        files2 = []
        classification_year = []
        channels_array = []
        gc.collect()
        
        for parameter in CHANNELS4:
            f = open(GRIB_FOLDER + parameter + "/" + str(year) + '.grib')
            files4.append(f)
            
        for parameter in CHANNELS2:
            f = open(GRIB_FOLDER + parameter + "/" + str(year) + '.grib')
            files2.append(f)
            
        isOdd = True
            
        while 1:
            gids = []
            for f in files4:
                gid = codes_grib_new_from_file(f)
                gids.append(gid)

            if gids[0] is None:
                break

            channels = np.empty((len(CHANNELS2) + len(CHANNELS4)))
            gidIt = 0
            for gid in gids:
                nearest = codes_grib_find_nearest(gid, latitude, longitude)[0]
                channels[gidIt] = nearest.value
                gidIt = gidIt + 1
                codes_release(gid)
                
            if (isOdd):
                for f in files2:
                    
                    gid = codes_grib_new_from_file(f)
                    nearest = codes_grib_find_nearest(gid, latitude, longitude)[0]
                    channels[gidIt] = nearest.value
                    gidIt = gidIt + 1
                    codes_release(gid)
                isOdd = False    
            else:
                for f in files2:
                    channels[gidIt] = channels_array[-1][gidIt]
                    gidIt = gidIt + 1
                isOdd = True

            channels_array.append(channels)
            classification_year.append(classify_step(channels))  

        for f in files4:
            f.close()
        for f in files2:
            f.close()
        return classification_year

    
def classify_step(channels):
    ''' 
        classifies the weather for a given step
        channels[0] is cloud cover (0-1)
        channels[1] is temperature (Kelvin)
        channels[2] is precipitation (Meter)
    '''
    
    if (channels[2] < 0.0001):
        # no prec
        if (channels[0] < 0.01):
            return 0
        if (channels[0] < 0.3):
            return 1
        if (channels[0] < 0.6):
            return 2
        return 3
    if (channels[1] < 275):
        # prec + freezing = snow
        return 6
    if (channels[2] < 0.005):
        return 4
    return 5
    
def resolve_classification(classification):
    if (classification == 0):
        return 'clear sky'
    if (classification == 1):
        return 'few clouds'
    if (classification == 2):
        return 'scattered clouds'
    if (classification == 3):
        return 'overcast clouds'
    if (classification == 4):
        return 'light rain'
    if (classification == 5):
        return 'heavy rain'
    if (classification == 6):
        return 'snow'
    if (classification == 7):
        return 'thunderstorm'
    if (classification == 8):
        return 'mist'
    return 'undef'
    
def process_classification(classification, errors):
    classification = np.asarray(classification)
    classification = classification[(classification.shape[0]-errors.shape[0]):]
  
    # prepare error list
    error_list = []
    error_array = np.empty((3, 2))
    error_array[:] = np.NAN
    for i in range(9):
        error_list.append([])
        
    for i in range(errors.shape[0]):
        error_list[classification[i]].append(errors[i])
        
    tuple_list = []
    for i in range(len(error_list)):
        tuple_list.append((i, len(error_list[i])))

    tuple_list.sort(key=lambda x: x[1], reverse=True)
    print(tuple_list)
    
    for i in range(error_array.shape[0]):
        e_list = error_list[tuple_list[i][0]]
        error_array[i, 1] = len(e_list)
        
        if (len(e_list) > 0):
            error_array[i, 0] = sum(e_list) / len(e_list)

    return error_array
    
def overall_classification(modelFile, location):
    '''
        classifies the weather for TEST_YEARS
        for all locations in LATLON, 
        returns an array with
        0: locations
        1: weather type (index 0-2) for the three most important ones
        2: the mean error and the amount (0: mean error, 1: amount)
    '''
    
    model = load_model(GRIB_FOLDER + 'data/' + location + '/' + modelFile + '/model/model.h5')
    
    classifications = np.empty((len(LATLON), 3, 2))
    for locIt in range(len(LATLON)):
        lat = LATLON[locIt][0]
        lon = LATLON[locIt][1]
        
        # the classification
        classification = classify_weather(lat, lon)
        
        # the errors
        train_params = get_default_data_params()
        train_params.lat = lat
        train_params.lon = lon

        testData = DatasetSquareArea(train_params)

        _, dataY = testData.inverse_transform_data()
        predict = testData.predict_data(model)
            
        # we're only interested in temperature, therefore the 0 channel.
        error = np.abs(dataY[:,0] - predict[:,0])
        
        classifications[locIt, :, :] = process_classification(classification, error)
        
    np.save(GRIB_FOLDER + 'data/classification/' + location + '_' + modelFile + '.npy', classifications)

def main():
    overall_classification(modelFile='n64_pTempSpress_mAll_f0_s20_l2', location='los_angeles')
    overall_classification(modelFile='n64_pTempSpress_mAll_f0_s20_l2', location='zurich')
    overall_classification(modelFile='n64_pTempSpress_mAll_f0_s20_l2', location='paris')
    return 1

if __name__ == "__main__":
    sys.exit(main())
