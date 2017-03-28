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
    train_params.steps_before = 8
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
        model = create_model(train_params.steps_before, nb_features, nb_features * 10)
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
    train_params.steps_before = 8
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
        model = create_model(train_params.steps_before, trainData.params.nb_features, nb_neurons)
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
    
def test_steps_before():
    ''' 
        trains a network with different window sizes
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

    steps_before_list = list(range(1, 20 + 1))

    results = np.zeros((len(steps_before_list), 2))
    results_it = 0

    #train and save the model files
    print ('training started...')
    for steps_before in steps_before_list:
        
        # load data
        train_params.steps_before = steps_before
        train_params.years = [2000,2001,2002]
        trainData = DatasetArea(train_params)
        
        train_params.years = [2003]
        testData = DatasetArea(train_params)
       
        model_file = 'test_steps_before/model_ws_' + str(steps_before) + '.h5'
        
        # create and fit the LSTM network
        print('creating model for steps_before: ' + str(steps_before))
        model = create_model(train_params.steps_before, trainData.params.nb_features, 100)
        train_model(model, trainData, epoch_count)
        model.save(model_file)
 
        print('testing model for steps_before: ' + str(steps_before))
        avg_all = evaluate_model_score(model, testData)
        results[results_it] = (steps_before, avg_all)
        results_it = results_it + 1
        np.save('test_steps_before/results.npy', results)
       
    # plot
    plt.plot(results[:,0], results[:,1])
   
    plt.xlabel('Window Size')
    plt.ylabel('RMSE (Kelvin)')
    plt.title('Window Size')
    plt.grid(True)
    plt.savefig("test_steps_before/plot.png")
    plt.show()
    
def test_network_depth():
    ''' 
        trains several networks with different number of LSTM layers
    '''
    train_params = AttrDict()
    train_params.steps_before = 12
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.npoints = 25
    train_params.grib_folder = '/media/isa/VIS1/temperature/'
    train_params.months = [6,7,8]

    network_depth_list = [1, 2, 3, 4]
    results = np.zeros((len(network_depth_list), 2))
    results_it = 0
    
    # load data
    train_params.years = [2000,2001,2002]
    trainData = DatasetNearest(train_params)
    
    train_params.years = [2003]
    testData = DatasetNearest(train_params)

    #train and save the model files
    print ('training started...')
    for network_depth in network_depth_list:

        model_file = 'test_network_depth/model_nd_' + str(network_depth) + '.h5'
        
        # create and fit the LSTM network
        print('creating model for network_depth: ' + str(network_depth))
        model = create_deep_model(train_params.steps_before, trainData.params.nb_features, train_params.npoints, network_depth)
        train_model(model, trainData, 50 * network_depth)
        model.save(model_file)
 
        print('testing model for network_depth: ' + str(network_depth))
        avg_all = evaluate_model_score(model, testData)
        results[results_it] = (network_depth, avg_all)
        results_it = results_it + 1
        np.save('test_network_depth/results.npy', results)
       
    # plot
    fig, ax = plt.subplots()
    width = 0.35
    ax.bar(results[:,0], results[:,1], width, color='b')
    ax.set_xticks(np.arange(1, 5) + width / 2)
    ax.set_xticklabels(('1', '2', '3', '4'))
    
    plt.xlabel('Network Depth')
    plt.ylabel('RMSE (Kelvin)')
    plt.ylim([0.6, 0.85])
    plt.title('Network Depth')
    plt.savefig("test_network_depth/plot.png")
    plt.show()
    
def test_network_activation():
    ''' 
        trains several networks with different activation layers
    '''
    train_params = AttrDict()
    train_params.steps_before = 12
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.npoints = 25
    train_params.grib_folder = '/media/isa/VIS1/temperature/'
    train_params.months = [6,7,8]

    network_activation_list = ('softmax', 'elu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear')
    results = np.zeros((len(network_activation_list)))
    results_it = 0
    
    # load data
    train_params.years = [2000,2001,2002]
    trainData = DatasetNearest(train_params)
    
    train_params.years = [2003]
    testData = DatasetNearest(train_params)

    #train and save the model files
    print ('training started...')
    for network_activation in network_activation_list:

        model_file = 'test_network_activation/model_na_' + network_activation + '.h5'
        
        # create and fit the LSTM network
        print('creating model for network_activation: ' + network_activation)
        model = create_model_activation(train_params.steps_before, trainData.params.nb_features, train_params.npoints, network_activation)
        train_model(model, trainData, 20)
        model.save(model_file)
 
        print('testing model for network_activation: ' + network_activation)
        avg_all = evaluate_model_score(model, testData)
        results[results_it] = avg_all
        results_it = results_it + 1
        np.save('test_network_activation/results.npy', results)
       
    # plot
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6, forward=True)
    width = 0.35
    ind = np.arange(len(network_activation_list))
    rects = ax.bar(ind, results, width, color='b')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(network_activation_list)

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % height,
                ha='center', va='bottom')
    
    plt.xlabel('Network Activation')
    plt.ylabel('RMSE (Kelvin)')
    plt.ylim([0.6, 0.9])
    plt.title('Network Activation')
    plt.savefig("test_network_activation/plot2.png")
    plt.show()
    
def test_forecast_distance():
    ''' 
        testing how well the network can predict
        n >= 1 steps into the future
        compare: recursive vs model
    '''
    train_params = AttrDict()
    train_params.steps_before = 30
    train_params.lat = 47.25
    train_params.lon = 189.0
    train_params.npoints = 25
    train_params.grib_folder = '/media/isa/VIS1/temperature/'
    train_params.months = [1,2,3,4,5,6,7,8,9,10,11,12]
    train_params.years = [2000,2001,2002]

    forecast_distance_list = [1,2,3,4,5,6,7,8]

    #train and save the model files
    print ('training started...')
    for forecast_distance in forecast_distance_list:

        model_file = 'test_forecast_distance_2/model_fd_' + str(forecast_distance) + '.h5'
        train_params.forecast_distance = forecast_distance
        trainData = DatasetNearest(train_params)
        
        # create and fit the LSTM network
        print('creating model for forecast_distance: ' + str(forecast_distance))
        model = create_model(train_params.steps_before, trainData.params.nb_features, train_params.npoints * 2)
        train_model(model, trainData, 20)
        model.save(model_file)
        
def evaluate_forecast_distance():
    ''' 
        testing how well the network can predict
        n >= 1 steps into the future
        compare: recursive vs model
    '''
    steps_before = 30
    test_params = AttrDict()
    test_params.steps_before = steps_before
    test_params.lat = 47.25
    test_params.lon = 189.0
    test_params.npoints = 25
    test_params.grib_folder = '/media/isa/VIS1/temperature/'
    test_params.months = [1]
    test_params.years = [2003]

    results_count = 10
    forecast_distance_list = [1,2,3,4,5,6,7,8]
    results = np.empty((len(forecast_distance_list), results_count, 2))
    window_before = np.empty((results_count, steps_before))

    # evaluate how the individual models do
    it = 0
    for forecast_distance in forecast_distance_list:

        model_file = 'test_forecast_distance_2/model_fd_' + str(forecast_distance) + '.h5'
        test_params.forecast_distance = forecast_distance
        dataset = DatasetNearest(test_params)
        
        # load the trained model
        model = load_model(model_file)
        
        # make predictions
        predict = model.predict(dataset.dataX)
   
        # invert predictions
        predict = dataset.scaler.inverse_transform(predict)
        dataY = dataset.scaler.inverse_transform(dataset.dataY)
        
        results[it,:,0] = predict[:results_count, 0]
        results[it,:,1] = dataY[:results_count, 0]
        it = it + 1
        
        # invert dataX
        dataX = np.empty(dataset.dataX.shape)
        for i in range(dataset.dataX.shape[0]):
            dataX[i] = dataset.scaler.inverse_transform(dataset.dataX[i])
            
        window_before[:] = dataX[:results_count, :, 0]
     
    np.save('test_forecast_distance_2/window_before.npy', window_before)
    np.save('test_forecast_distance_2/results.npy', results)
     
    # evaluate how the single model does
    model_file = 'test_forecast_distance_2/model_fd_1.h5'
    test_params.forecast_distance = 1
    dataset = DatasetNearest(test_params)
        
    # load the trained model
    model = load_model(model_file)
    predict_m = predict_multiple(model, dataset, forecast_distance_list[-1])
    for i in range(predict_m.shape[0]):
        predict_m[i] = dataset.scaler.inverse_transform(predict_m[i])

    # plot
    nan_array = np.empty((steps_before - 1))
    nan_array.fill(np.nan)
    nan_array2 = np.empty((len(forecast_distance_list)))
    nan_array2.fill(np.nan)
    ind = np.arange(steps_before + len(forecast_distance_list))
    
    for i in range(results.shape[1]):
        plt.cla()
        fig, ax = plt.subplots()

        forecasts = np.concatenate((nan_array, window_before[i, -1:], results[:, i, 0]))
        forecasts_multiple = np.concatenate((nan_array, window_before[i, -1:], predict_m[i, :, 0]))
        ground_truth = np.concatenate((nan_array, window_before[i, -1:], results[:, i, 1]))
        network_input = np.concatenate((window_before[i, :], nan_array2))
     
        ax.plot(ind, network_input, 'b-x', label='Network input')
        ax.plot(ind, forecasts_multiple, 'm-x', label='Recursive model forecast')
        ax.plot(ind, forecasts, 'r-x', label='Individual model forecast')
        ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')

        start_date = dataset.frames_data[dataset.frames_idx[i + steps_before]]
        end_date = dataset.frames_data[dataset.frames_idx[i + steps_before + dataset.params.forecast_distance - 1]]
        
        start_date_s = start_date.date + '-' + start_date.time
        end_date_s = end_date.date + '-' + end_date.time
        
        plt.xlabel('Time (6h steps)')
        plt.ylabel('Temperature (Kelvin)')
        plt.title('Multiple Steps Forecast (' + start_date_s + ' -- ' + end_date_s + ')')
        plt.legend(loc='upper right')
        plt.savefig("test_forecast_distance_2/plot" + str(i) + ".png")
        
def main():
    #test_nb_features()
    #test_nb_neurons()
    #test_steps_before()
    #test_network_depth()
    #test_network_activation()
    #test_forecast_distance()
    #evaluate_forecast_distance()
    #draw_fd_plot()
    #test_many_to_many()
    return 1

if __name__ == "__main__":
    sys.exit(main())
