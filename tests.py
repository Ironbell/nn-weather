from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from dataset import *
from network import *
from visualise import *

def test_nb_features():
    ''' 
        trains a network with different number of features 
        compares and visualises the prediction error
    '''
    train_params = AttrDict()
    train_params.window_size = 8
    train_params.grib_folder = '/media/isa/VIS1/temperature/'
    train_params.months = [6,7,8]
    epoch_count = 10

    nb_features_list = [1, 3, 5, 9, 16, 25, 40]

    results = np.zeros((len(nb_features_list), 3))
    results_it = 0
    
    #train and save the model files
    print ('training started...')
    for nb_features in nb_features_list:
       
        train_params.lat = 47.25
        train_params.lon = 189.0
        train_params.npoints = nb_features
        train_params.years = [2000,2001,2002]

        # load the data from the .grib files
        trainData = DatasetNearest(train_params)
        model_file = 'test_nb_features/model_features_' + str(nb_features) + '.h5'
        
        # create and fit the LSTM network
        print('creating model for features: ' + str(nb_features))
        model = create_model(train_params.window_size, nb_features, nb_features * 10)
        train_model(model, trainData, epoch_count)
        model.save(model_file)
        
        # now evaluate
        train_params.years = [2003]
        testData = DatasetNearest(train_params)
        
        print('testing model for features: ' + str(nb_features))
        avg_1, avg_all = evaluate_model_score_2(model, testData)
        results[results_it] = (nb_features, avg_1, avg_all)
        results_it = results_it + 1
        np.save('test_nb_features/results.npy', results)
       
    # plot
    plt.plot(results[:,0], results[:,1], label='Avg error of central feature')
    plt.plot(results[:,0], results[:,2], label='Avg error of all features')

    plt.xlabel('# Features')
    plt.ylabel('RMSE (Kelvin)')
    plt.title('Number of Features')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig("test_nb_features/plot.png")
    plt.show()

def test_nb_neurons():
    ''' 
        trains a network with different number neurons 
        compares and visualises the prediction error
    '''
    train_params = AttrDict()
    train_params.window_size = 8
    train_params.start_lat = 45.839
    train_params.end_lat = 47.74
    train_params.start_lon = 186.10
    train_params.end_lon = 190.52
    train_params.grib_folder = '/media/isa/VIS1/temperature/'
    train_params.months = [6,7,8]
    
    epoch_count = 20
    
    # load data
    train_params.years = [2000,2001,2002]
    trainData = DatasetArea(train_params)
    
    train_params.years = [2003]
    testData = DatasetArea(train_params)

    nb_neurons_list = [ 2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 ]

    results = np.zeros((len(nb_neurons_list), 2))
    results_it = 0

    #train and save the model files
    print ('training started...')
    for nb_neurons in nb_neurons_list:
       
        model_file = 'test_nb_neurons/model_neurons_' + str(nb_neurons) + '.h5'
        
        # create and fit the LSTM network
        print('creating model for neurons: ' + str(nb_neurons))
        model = create_model(train_params.window_size, trainData.vector_size, nb_neurons)
        train_model(model, trainData, epoch_count)
        model.save(model_file)
 
        print('testing model for neurons: ' + str(nb_neurons))
        avg_all = evaluate_model_score(model, testData)
        results[results_it] = (nb_neurons, avg_all)
        results_it = results_it + 1
        np.save('test_nb_neurons/results.npy', results)
       
    # plot
    plt.plot(results[:,0], results[:,1])
   
    plt.xlabel('# Neurons')
    plt.ylabel('RMSE (Kelvin)')
    plt.title('Number of Neurons')
    plt.grid(True)
    plt.savefig("test_nb_neurons/plot.png")
    plt.show()

def test_window_size():
    ''' 
        trains a network with different number neurons 
        compares and visualises the prediction error
    '''
    train_params = AttrDict()
    train_params.start_lat = 45.839
    train_params.end_lat = 47.74
    train_params.start_lon = 186.10
    train_params.end_lon = 190.52
    train_params.grib_folder = '/media/isa/VIS1/temperature/'
    train_params.months = [6,7,8]
    
    epoch_count = 30

    window_size_list = list(range(1, 20 + 1))

    results = np.zeros((len(window_size_list), 2))
    results_it = 0

    #train and save the model files
    print ('training started...')
    for window_size in window_size_list:
        
        # load data
        train_params.window_size = window_size
        train_params.years = [2000,2001,2002]
        trainData = DatasetArea(train_params)
        
        train_params.years = [2003]
        testData = DatasetArea(train_params)
       
        model_file = 'test_window_size/model_ws_' + str(window_size) + '.h5'
        
        # create and fit the LSTM network
        print('creating model for window_size: ' + str(window_size))
        model = create_model(train_params.window_size, trainData.vector_size, 100)
        train_model(model, trainData, epoch_count)
        model.save(model_file)
 
        print('testing model for window_size: ' + str(window_size))
        avg_all = evaluate_model_score(model, testData)
        results[results_it] = (window_size, avg_all)
        results_it = results_it + 1
        np.save('test_window_size/results.npy', results)
       
    # plot
    plt.plot(results[:,0], results[:,1])
   
    plt.xlabel('Window Size')
    plt.ylabel('RMSE (Kelvin)')
    plt.title('Window Size')
    plt.grid(True)
    plt.savefig("test_window_size/plot.png")
    plt.show()

def main():
    #test_nb_features()
    #test_nb_neurons()
    test_window_size()
    return 1

if __name__ == "__main__":
    sys.exit(main())
