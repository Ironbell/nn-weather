from attrdict import AttrDict
from keras.models import load_model
import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt

import os

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from attrdict import AttrDict

from eccodes import *

def create_dataset():
    """
        Creates a dataset with 
        dataX and dataY for training/testing
        with parameters according to the baseline paper.
    """    
    # load the temperature data from years 1999-2009
    # don't include the 29. feb
    # use the maximum from the 4 values per day (maxtemp)
    years = list(range(1999, 2009))
    longitude = -79.63 + 180
    latitude = 43.68
    grib_folder = '/media/isa/VIS1/temperature/'
    maxtemp_data = np.empty((365, len(years)))

    # load the data
    for year_it in range(len(years)):
        f = open(grib_folder + str(years[year_it]) + '.grib')
        time_it = 0
        day_it = 0
        day_data = np.empty((4))
        
        while 1:
            gid = codes_grib_new_from_file(f)
            if (gid is None):
                break
            # check for feb 29
            dataDate = str(codes_get(gid, 'dataDate')).zfill(8)
            if (dataDate[4:6] == '02' and dataDate[6:8] == '29'):
                codes_release(gid)
                print ('skipping ' + dataDate)
                continue

            nearest = codes_grib_find_nearest(gid, latitude, longitude)[0]
            
            if (nearest.value == codes_get_double(gid, 'missingValue')):
                raise Warning('missing value!')
                
            day_data[time_it] = nearest.value
            
            time_it = time_it + 1
            if time_it == 4:
                maxtemp_data[day_it, year_it] = day_data.max()
                day_it = day_it + 1
                time_it = 0
                
            codes_release(gid)
    
        f.close()
        
    min_data, ptp_data = maxtemp_data.min(), maxtemp_data.ptp()
    maxtemp_data = (maxtemp_data - min_data) / ptp_data
        
    # x: [year0, year1, ...., year9]
    # y: the next day, ie year9 + 1 day
    # get x data: x is all samples but the last (it wouldn't have an corresponding y)
    data_x = maxtemp_data[:-1, :]
    # get y data: y is the next day
    data_y = np.empty((364))
    for d in range(364):
        data_y[d] = maxtemp_data[d + 1, -1]

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)
    
    dataset = AttrDict()
    dataset.train_x = train_x
    dataset.test_x = test_x
    dataset.train_y = train_y
    dataset.test_y = test_y
    dataset.min_data = min_data
    dataset.ptp_data = ptp_data
    return dataset
    

def create_model(nb_layers, nb_neurons, nb_features=10, activation_function='linear'):
    """ 
        creates, compiles and returns a ANN model 
        @param nb_layers: how many hidden layers should be used (all dense)
        @param nb_neurons: how many neurons per hidden layer
        @param nb_features: how many input features (there are ten in the paper)
        @param activation_function: according to the paper, this is 'linear' or 'sigmoid'
    """
    #DROPOUT = 0.5 # the paper doesn't use dropout (bad!!!)

    model = Sequential() 
    model.add(Dense(input_dim=nb_features, output_dim=nb_neurons))    
    for _ in range(nb_layers):
        model.add(Dense(nb_neurons))
    model.add(Dense(1))
    model.add(Activation(activation_function)) 
    
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])  
    return model

def train_model(model, dataX, dataY, nb_epoch, model_folder):
    """ 
        trains the maxtemp model and saves
    """
    history = model.fit(dataX, dataY, batch_size=1, nb_epoch=nb_epoch, validation_split=0.3)
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # save model
    model.save(model_folder + '/model.h5')
  
    # plot training and val loss/accuracy
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_folder + '/history_acc.png')
    plt.cla()    

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(model_folder + '/history_loss.png')
    plt.cla()
    
def evaluate_model(model, dataset):
    """
        testing how big the MSE is for our test data
        @param model: the model to test
        @param dataset: dataset with test_x, test_y and scaler
        @return: the MSE between prediction and ground truth
    """
    predict = model.predict(dataset.test_x)
    
    predict = predict * dataset.ptp_data + dataset.min_data
    test_y = dataset.test_y * dataset.ptp_data + dataset.min_data
    
    mse = mean_squared_error(test_y, predict)
    return mse

def main():
    subfolder = 'test_baseline_maxtemp/'
    dataset = create_dataset()
    
    # different number of layers:
    nb_layers_list = [1, 5, 10]
    # different number of neurons:
    nb_neurons_list = [20, 50, 80]
    # different activation functions:
    act_functions = ['linear', 'sigmoid']

    for nb_layers in nb_layers_list:
        results = np.empty(((len(nb_neurons_list) * len(act_functions)), 3))
        n_it = 0
        for nb_neurons in nb_neurons_list:
            act_it = 0
            for act_func in act_functions:
                model = create_model(nb_layers=nb_layers, nb_neurons=int(nb_neurons/nb_layers), activation_function=act_func)
                
                model_folder = subfolder + 'model_' + str(nb_layers) + '_' + str(nb_neurons) + '_' + act_func
                train_model(model, dataset.train_x, dataset.train_y, nb_epoch=50, model_folder=model_folder)
                
                mse = evaluate_model(model, dataset)
                results[len(act_functions) * n_it + act_it, 0] = int(nb_neurons/nb_layers)
                results[len(act_functions) * n_it + act_it, 1] = act_it
                results[len(act_functions) * n_it + act_it, 2] = mse

                act_it = act_it + 1
            
            n_it = n_it + 1

        fig=plt.figure()
        ax = fig.add_subplot(111)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        colLabels=('Neurons/Layer', 'Activation Function', 'MSE')
        the_table = ax.table(cellText=results,
              colLabels=colLabels,
              loc='center')
        plt.savefig(subfolder + 'table_layers_' + str(nb_layers) + '.png')
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
