import os
from attrdict import AttrDict
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from dataset import *
from network import *
from visualise import *
from eccodes import *

import colormap as cm
import time

GRIB_FOLDER = '/media/isa/VIS1/'
MODEL_PATH = 'data/los_angeles/n64_pTempSpress_mAll_f0_s20_l2/model/model.h5'
RADIUS = 3

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
    params.location = ''
    params.grib_parameters = ['temperature', 'surface_pressure']
    params.months = list(range(1, 12 + 1))
    params.years = [2016]
    params.hours = [0, 6, 12, 18]
    return params
    
def load_cached_data(data_params):
    '''
        only one year possible for now.
    ''' 
    all_frames = []
    
    for parameter in data_params.grib_parameters:
        f = open(data_params.grib_folder + parameter + "/" + str(data_params.years[0]) + '.grib')
        frames = [] 
        
        while 1:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
     
            iterid = codes_grib_iterator_new(gid, 0)
                
            frame = np.empty([241,480], dtype=float)

            while 1:
                result = codes_grib_iterator_next(iterid)
                if not result:
                    break
     
                [lat, lon, value] = result
                frame[lonlat_to_idx(lon, lat)] = max(0, value)

            frames.append(frame)
            codes_grib_iterator_delete(iterid)
            codes_release(gid)
        
        arrayed_frames = np.asarray(frames)
        gc.collect()
        print ('frames shape:')
        print (arrayed_frames.shape)
        all_frames.append(arrayed_frames)

        f.close()
        
    return all_frames

def run_overall_testing():
    ''' 
        runs a test for the visualisation model.
    '''
    subfolder = GRIB_FOLDER + 'data/overall_testing/'
    
    model = load_model(GRIB_FOLDER + MODEL_PATH)
    train_params = get_default_data_params()
    channels = len(train_params.grib_parameters)
    
    cached_data = load_cached_data(train_params)
    
    score = np.empty((240, 480, channels, 4))
    score[:] = np.NAN
    try:
        score = np.load(subfolder + 'score.npy')
        assert score.shape == (240, 480, channels, 4)
        print('score loaded')
        
    except IOError:
        print('creating score anew')  

   
    latIt = 0
    lonIt = 0
    for lat in drange(-90, 90, GRID_SIZE):
        lonIt = 0
        for lon in drange(0, 360, GRID_SIZE):

            if not np.isnan(score[latIt, lonIt, 0, 0]):
                lonIt = lonIt + 1
                continue 
            gc.collect()
            train_params.lat = lat
            train_params.lon = lon

            testData = DatasetSquareAreaCached(train_params, cached_data)

            _, dataY = testData.inverse_transform_data()
            predict = testData.predict_data(model)
            
            for i in range(dataY.shape[1]): # loop over channels
                error = dataY[:,i] - predict[:,i]
                score[latIt, lonIt, i, 0] = np.mean(error) # ME
                score[latIt, lonIt, i, 1] = np.std(error) # STD ME
                score[latIt, lonIt, i, 2] = np.mean(np.absolute(error)) # MAE
                score[latIt, lonIt, i, 3] = np.std(np.absolute(error)) # STD MAE

            np.save(subfolder + 'score.npy', score)
            print('lat: ' + str(lat) + ', lon: ' + str(lon) + ', MAE: ' + str(score[latIt, lonIt, 0, 2]))
            
            lonIt = lonIt + 1
            
        latIt = latIt + 1

def plot_overall_testing():
    ''' 
        generates plots for ME, STD ME, MAE and STD MAE
    '''
    score = np.load(GRIB_FOLDER + 'data/overall_testing/score.npy')
    score = np.flipud(score)
    colmap = cm.RdYlGn_r(100)
    extent = [0,359.25,-90,89.25]

    # ME
    
    plt.imshow(score[:,:,0,0], cmap=colmap, interpolation='none', extent=extent)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Mean Error Overview (2016)')
    
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label('Mean Error (Kelvin)', rotation=270, labelpad=20)
    plt.savefig(GRIB_FOLDER + 'data/overall_testing/me.png')
    plt.close('all')

    # STD ME

    plt.imshow(score[:,:,0,1], cmap=colmap, interpolation='none', extent=extent)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Std Mean Error Overview (2016)')
    
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label('Std Mean Error (Kelvin)', rotation=270, labelpad=20)
    plt.savefig(GRIB_FOLDER + 'data/overall_testing/stdme.png')
    plt.close('all')

    # MAE

    plt.imshow(score[:,:,0,2], cmap=colmap, interpolation='none', extent=extent)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Mean Absolute Error Overview (2016)')
    
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label('Mean Absolute Error (Kelvin)', rotation=270, labelpad=20)
    plt.savefig(GRIB_FOLDER + 'data/overall_testing/mae.png')
    plt.close('all')

    # STD MAE

    plt.imshow(score[:,:,0,3], cmap=colmap, interpolation='none', extent=extent)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Std Mean Absolute Error Overview (2016)')
    
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label('Std Mean Absolute Error (Kelvin)', rotation=270, labelpad=20)
    plt.savefig(GRIB_FOLDER + 'data/overall_testing/stdmae.png')
    plt.close('all')


def main(): 
    run_overall_testing()
    plot_overall_testing()
    return 1

if __name__ == "__main__":
    sys.exit(main())
