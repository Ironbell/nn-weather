from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout  
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib as mpl

from dataset import *
import colormap as cm

def create_model(window_size, feature_count):
    """ 
        creates, compiles and returns a RNN model 
        @param window_size: the number of previous time steps
        @param feature_count: the number of features in the model
    """
    hidden_neurons = 100
    dropout = 0.2
    
    model = Sequential()  
    model.add(LSTM(hidden_neurons, input_shape=(window_size, feature_count), return_sequences=False))  
    model.add(Dropout(dropout))
    #model.add(LSTM(hidden_neurons, return_sequences=False))
    #model.add(Dropout(dropout))
    model.add(Dense(feature_count))  
    model.add(Activation("linear"))   
    
    model.compile(loss="mean_squared_error", optimizer="rmsprop")  
    return model

def train_model(model, dataset, epoch_count):
    """ 
        trains a model given the dataset to train on
        @param model: the model to train
        @param dataset: the dataset to train the model on
        @param epoch_count: number of epochs to train
        
        TODO: maybe specify if the model needs to be saved between epochs?
        TODO: maybe specify a target validation loss?
    """
    model.fit(dataset.trainX, dataset.trainY, batch_size=1, nb_epoch=epoch_count, validation_split=0.05)
    
def evaluate_model(model, dataset):
    """ 
        evaluates the model given the dataset (training and test data)
        @param model: the model to evaluate_model
        @param dataset: training and test data to evaluate
    """
    # make predictions
    trainPredict = model.predict(dataset.trainX)
    testPredict = model.predict(dataset.testX)
    
    # invert predictions
    trainPredict = dataset.scaler.inverse_transform(trainPredict)
    dataset.trainY = dataset.scaler.inverse_transform(dataset.trainY)
    testPredict = dataset.scaler.inverse_transform(testPredict)
    dataset.testY = dataset.scaler.inverse_transform(dataset.testY)
    
    for i in range(dataset.trainX.shape[0]):
        dataset.trainX[i] = dataset.scaler.inverse_transform(dataset.trainX[i])
    for i in range(dataset.testX.shape[0]):
        dataset.testX[i] = dataset.scaler.inverse_transform(dataset.testX[i])
    
    # calculate root mean squared error
    scores = []
    for i in range(dataset.trainY.shape[1]):
        scores.append(math.sqrt(mean_squared_error(dataset.trainY[:,i], trainPredict[:,i])))
    
    avg = sum(scores)/len(scores)
    print('Train Score average: %.2f RMSE' % avg)
        
    scores = []
    for i in range(dataset.testY.shape[1]):
        scores.append(math.sqrt(mean_squared_error(dataset.testY[:,i], testPredict[:,i])))

    avg = sum(scores)/len(scores)
    print('Test Score average: %.2f RMSE' % avg)    

    return trainPredict, testPredict

def plot_predictions(dataset, trainPredict, testPredict):
    window_size = dataset.params.window_size
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset.frames)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[window_size:len(trainPredict) + window_size, :] = trainPredict[:,:]
    
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset.frames)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (window_size * 2) + 1:len(dataset.frames)-1, :] = testPredict[:,:]
    
    # plot baseline and predictions
    plt.plot(dataset.scaler.inverse_transform(dataset.frames))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    
def plot_predictions_images(params, file_name, dataX, dataY, predict):
    nb_frames = 10
    window_size = params.window_size
    
    # calculate lat and lon for axis scaling
    nelat = round_nearest(params.end_lat, GRID_SIZE)
    nslat = round_nearest(params.start_lat, GRID_SIZE)
    nelon = round_nearest(params.end_lon, GRID_SIZE)
    nslon = round_nearest(params.start_lon, GRID_SIZE)

    lat_range = int((1 + (nelat - nslat) / GRID_SIZE))
    lon_range = int((1 + (nelon - nslon) / GRID_SIZE))

    for i in range(nb_frames):
        fig, axes = plt.subplots(nrows=1, ncols=window_size + 2)
        fig.set_size_inches((window_size + 2) * 6, 5, forward=True)
        
        # plot trajectory
        for axis_nr in range(window_size):
            ax = axes.flat[axis_nr]
            ax.text(nslon + 0.5, nslat + 0.25, 'Inital trajectory', fontsize=10, color='w')
            toplot = np.reshape(dataX[i,axis_nr,:], (lat_range, -1))
            ax.tick_params(labelsize=6)
            im = ax.imshow(toplot, cmap=cm.YlOrRd(), extent=[nslon,nelon,nslat,nelat])
        
        # plot prediction
        ax = axes.flat[window_size]
        ax.text(nslon + 0.5, nslat + 0.25, 'Prediction', fontsize=10, color='w')
        toplot = np.reshape(predict[i,:], (lat_range, -1))
        ax.tick_params(labelsize=6)
        im = ax.imshow(toplot, cmap=cm.YlOrRd(), extent=[nslon,nelon,nslat,nelat])
        
        # plot ground truth 
        ax = axes.flat[window_size + 1]
        plt.text(nslon + 0.5, nslat + 0.25, 'Ground truth', fontsize=10, color='w')
        toplot = np.reshape(dataY[i,:], (lat_range, -1))
        ax.tick_params(labelsize=6)
        im = ax.imshow(toplot, cmap=cm.YlOrRd(), extent=[nslon,nelon,nslat,nelat])

        cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], shrink=0.6, pad=0.02)
        plt.colorbar(im, cax=cax, **kw)
       
        '''
        ax = fig.add_subplot(133)
        plt.text(1, 3, 'Error', fontsize=20)
        toplot3 = abs(toplot1 - toplot2) 
        plt.imshow(toplot3)
        plt.colorbar()'''
        
        plt.savefig(('%i_animate_' % (i + 1)) + file_name + '.png', bbox_inches='tight',dpi=100) 
    
def plot_predictions_images_test(dataset, testPredict):
    plot_predictions_images(dataset.params, \
    'test', dataset.testX, dataset.testY, testPredict)

def plot_predictions_images_train(dataset, trainPredict):
    plot_predictions_images(dataset.params, \
    'train', dataset.trainX, dataset.trainY, trainPredict)
