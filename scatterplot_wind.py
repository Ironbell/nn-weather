import os, sys, gc
import numpy as np
import matplotlib.pyplot as plt

import colormap as cm

GRIB_FOLDER = '/media/isa/VIS1/'
TEST_YEARS = list(range(1990, 2016 + 1))
TEST_MONTHS = list(range(1, 12 + 1))
RADIUS = 3
CHANNEL = 0

def scatterplot(loc='zurich', x_axis='wind_u', y_axis='wind_v', model='n64_pTempSpress_mAll_f0_s20_l2'):
    ''' 
        plots two grib attributes in a scatterplot for a given location
    '''
    model_folder = '/media/isa/VIS1/data/zurich/' + model + '/'
    subfolder = GRIB_FOLDER + 'data/' + loc + '/scatterplot/'
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
        
    x = []
    y = []
    errors = []

    for year in TEST_YEARS:
        for month in TEST_MONTHS:
            gc.collect()
            
            x_array = np.load(GRIB_FOLDER + x_axis + '/' + loc + '/' + str(year) + '/' + str(month) + '.npy')
            y_array = np.load(GRIB_FOLDER + y_axis + '/' + loc + '/' + str(year) + '/' + str(month) + '.npy')
            error_array = np.load(model_folder + str(year) + '/' + str(month).zfill(2) + '_error.npy')[:,:,CHANNEL]
            if (x_array[:,:,RADIUS,RADIUS].shape != error_array.shape):
                continue

            for day in range(x_array.shape[0]):
                for hour in range(x_array.shape[1]):
                    
                    if (error_array[day, hour] > 3.0):

                        x.append(x_array[day, hour, RADIUS, RADIUS])
                        y.append(y_array[day, hour, RADIUS, RADIUS])
                        errors.append(error_array[day, hour])

    x = np.asarray(x)
    y = np.asarray(y)
    errors = np.asarray(errors)
    #errors = np.clip(errors, 0, 5)
   
    #colors = np.random.rand(50)
    #area = np.pi * (15 * np.random.rand(50))**2  # 0 to 15 point radii

    #plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.scatter(x, y, c=errors, cmap=cm.Rd(100))
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.axhline(0, color=(0, 0, 0, 0.5))
    plt.axvline(0, color=(0, 0, 0, 0.5))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Mean Absolute Error (Kelvin)', rotation=270, labelpad=30)
    plt.savefig(subfolder + model + '_' + x_axis + '_' + y_axis + '.png', dpi=200)
    plt.close('all')

def main(): 
    scatterplot(loc='zurich', x_axis='wind_u', y_axis='wind_v', model='n64_pTempSpress_mAll_f0_s20_l2')
    scatterplot(loc='zurich', x_axis='wind_u', y_axis='wind_v', model='n64_pTempWuWv_mAll_f0_s20_l2')
    scatterplot(loc='zurich', x_axis='wind_u', y_axis='wind_v', model='n64_pTempFuFv_mAll_f0_s20_l2')
    scatterplot(loc='zurich', x_axis='wind_u', y_axis='wind_v', model='n64_pTempAvAu_mAll_f0_s20_l2')
    scatterplot(loc='zurich', x_axis='wind_u', y_axis='wind_v', model='n64_pTempFuFvAvAu_mAll_f0_s20_l2')
    return 1

if __name__ == "__main__":
    sys.exit(main())
