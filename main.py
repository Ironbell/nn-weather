﻿from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import json

from dataset import *
from network import *
from visualise import *

def train(config):
    # set (and check) params in config
    train_params = AttrDict()
    train_params.window_size = config['window_size']
    train_params.start_lat = config['start_lat']
    train_params.end_lat = config['end_lat']
    train_params.start_lon = config['start_lon']
    train_params.end_lon = config['end_lon']
    train_params.grib_folder = config['grib_folder']
    train_params.years = config['years']
    train_params.months = config['months']
    epoch_count = config['epoch_count']
    model_file = config['model_file']

    # load the data from the .grib files
    trainData = DatasetTrain(train_params)
    
    # create and fit the LSTM network
    model = create_model(train_params.window_size, trainData.vector_size)
    train_model(model, trainData, epoch_count)
    model.save(model_file)
    
def evaluate(config):
    # set (and check) params in config    
    test_params = AttrDict()
    test_params.window_size = config['window_size']
    test_params.start_lat = config['start_lat']
    test_params.end_lat = config['end_lat']
    test_params.start_lon = config['start_lon']
    test_params.end_lon = config['end_lon']
    test_params.grib_folder = config['grib_folder']
    test_params.years = config['years']
    test_params.months = config['months']
    max_frames = config['max_frames']
    
    model = load_model(config['model_file'])
    
    testData = DatasetTrain(test_params)
    
    #predict
    predict = evaluate_model(model, testData)
    plot_predictions_images(testData, predict, 'plots/', max_frames)
    
def display():
    score = np.load('score.npy')
    display_score(score)

def main():
    '''model = load_model('weather_model.h5')
    for weight in model.get_weights():
        print(weight.shape)
    return 1'''
    
    model = create_model(1, 28)
    for weight in model.get_weights():
        print(weight.shape)
    return 1
    
    config = json.loads(open(sys.argv[1]).read())
    print(config)

    if (config['action'] == 'train'):
        train(config)
    elif (config['action'] == 'evaluate'):
        evaluate(config)
    #elif (config['action'] == 'visualise'):
    #    visualise(config)

    return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py configuration.json")
        sys.exit(1)
    else:
        sys.exit(main())
