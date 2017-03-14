import matplotlib.pyplot as plt
import matplotlib as mpl

from dataset import *
import colormap as cm

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
        
        plt.savefig(file_name + ('%i_animate.png' % (i + 1)), bbox_inches='tight',dpi=100) 
    
def plot_predictions_images_test(dataset, testPredict):
    plot_predictions_images(dataset.params, \
    'plots/test/', dataset.testX, dataset.testY, testPredict)

def plot_predictions_images_train(dataset, trainPredict):
    plot_predictions_images(dataset.params, \
    'plots/train/', dataset.trainX, dataset.trainY, trainPredict)
