from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import json

from dataset import *
from network import *
from visualise import *

def train(config):
    # set (and check) params in config
    train_params = AttrDict()
    train_params.steps_before = config['steps_before']
    train_params.steps_after = config['steps_after']
    train_params.start_lat = config['start_lat']
    train_params.end_lat = config['end_lat']
    train_params.start_lon = config['start_lon']
    train_params.end_lon = config['end_lon']
    train_params.grib_folder = config['grib_folder']
    train_params.hours = config['hours']
    train_params.years = config['years']
    train_params.months = config['months']
    epoch_count = config['epoch_count']
    model_folder = config['model_folder']

    # load the data from the .grib files
    trainData = DatasetArea(train_params)
    
    # create and fit the LSTM network
    model = create_model(train_params.steps_before, train_params.steps_after, trainData.vector_size, 200)
    train_model(model, trainData, epoch_count, model_folder)
    
def evaluate(config):
    # set (and check) params in config    
    test_params = AttrDict()
    test_params.steps_before = config['steps_before']
    test_params.steps_after = config['steps_after']
    test_params.start_lat = config['start_lat']
    test_params.end_lat = config['end_lat']
    test_params.start_lon = config['start_lon']
    test_params.end_lon = config['end_lon']
    test_params.grib_folder = config['grib_folder']
    test_params.grib_parameters = config['grib_parameters']
    train_params.hours = config['hours']
    test_params.years = config['years']
    test_params.months = config['months']
    max_frames = config['max_frames']
    
    model = load_model(config['model_file'])
    
    testData = DatasetArea(test_params)
    
    #predict
    predict = evaluate_model(model, testData)
    plot_predictions_images(testData, predict, 'plots/', max_frames)
    
def display():
    score = np.load('score.npy')
    display_score(score)

def main():    
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
