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
    
def plot_predictions_images(dataset, predict, folder_name, max_frames):
    window_size = dataset.params.window_size
    
    # calculate lat and lon for axis scaling
    nelat = round_nearest(dataset.params.end_lat, GRID_SIZE)
    nslat = round_nearest(dataset.params.start_lat, GRID_SIZE)
    nelon = round_nearest(dataset.params.end_lon, GRID_SIZE)
    nslon = round_nearest(dataset.params.start_lon, GRID_SIZE)

    lat_range = int((1 + (nelat - nslat) / GRID_SIZE))
    lon_range = int((1 + (nelon - nslon) / GRID_SIZE))
    
    colmap = cm.YlOrRd()

    for i in range(min(max_frames, len(predict))):
        fig, axes = plt.subplots(nrows=1, ncols=window_size + 2)
        fig.set_size_inches((window_size + 2) * 6, 5, forward=True)
        
        # plot trajectory
        for axis_nr in range(window_size):
            ax = axes.flat[axis_nr]
            ax.text(nslon + 0.5, nslat + 0.25, 'Inital trajectory', fontsize=10, color='w')
            toplot = np.reshape(dataset.dataX[i,axis_nr,:], (lat_range, -1))
            ax.tick_params(labelsize=6)
            ax.set_title(dataset.frames_data[dataset.frames_idx[i + axis_nr]].date + '-' + \
                dataset.frames_data[dataset.frames_idx[i + axis_nr]].time)
            im = ax.imshow(toplot, cmap=colmap, extent=[nslon,nelon,nslat,nelat])
        
        # plot prediction
        ax = axes.flat[window_size]
        ax.text(nslon + 0.5, nslat + 0.25, 'Prediction', fontsize=10, color='w')
        toplot = np.reshape(predict[i,:], (lat_range, -1))
        ax.tick_params(labelsize=6)
        ax.set_title(dataset.frames_data[dataset.frames_idx[i + window_size]].date + '-' + \
            dataset.frames_data[dataset.frames_idx[i + window_size]].time)
        im = ax.imshow(toplot, cmap=colmap, extent=[nslon,nelon,nslat,nelat])
        
        # plot ground truth 
        ax = axes.flat[window_size + 1]
        plt.text(nslon + 0.5, nslat + 0.25, 'Ground truth', fontsize=10, color='w')
        toplot = np.reshape(dataset.dataY[i,:], (lat_range, -1))
        ax.tick_params(labelsize=6)
        ax.set_title(dataset.frames_data[dataset.frames_idx[i + window_size]].date + '-' + \
            dataset.frames_data[dataset.frames_idx[i + window_size]].time)
        im = ax.imshow(toplot, cmap=colmap, extent=[nslon,nelon,nslat,nelat])

        cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], shrink=0.6, pad=0.02)
        plt.colorbar(im, cax=cax, **kw)
       
        '''
        ax = fig.add_subplot(133)
        plt.text(1, 3, 'Error', fontsize=20)
        toplot3 = abs(toplot1 - toplot2) 
        plt.imshow(toplot3)
        plt.colorbar()'''
        
        plt.savefig(folder_name + ('%i_animate.png' % (i + 1)), bbox_inches='tight',dpi=100)   

def display_score(score):
    colmap = cm.RdYlGn_r(10)
    score = np.fliplr(score.transpose())
    
    plt.imshow(score, cmap=colmap, interpolation='none', extent=[2000,2016,1,12])

    plt.xlabel('Years')
    plt.ylabel('Months')
    plt.title('Error comparision over temperature data')
    plt.savefig("test.png")
    cb = plt.colorbar()
    cb.set_label('Avg RMSE (Kelvin)', rotation=270, labelpad=20)
    plt.show()

