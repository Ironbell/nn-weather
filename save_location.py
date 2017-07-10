import os, math, re, gc
import decimal
import numpy as np
import scipy as sp
import calendar

from attrdict import AttrDict

from eccodes import *

GRID_SIZE = 0.75
RADIUS = 3
GRIB_FOLDER = '/media/isa/VIS1/'

def drange(x, y, jump):
  x_ = decimal.Decimal(x)
  while x_ < y:
    yield float(x_)
    x_ += decimal.Decimal(jump)

# loc='zurich', latitude=47.25, longitude=8.25
# loc='paris', latitude=48.75, longitude=2.25
   
def save_location(grib_parameter, loc='los_angeles', latitude=33.75, longitude=241.5):
    """ 
        Loads data from the grib file and saves it for further faster use
    """
    years = list(range(1979, 2017))
    months = list(range(1, 13))
    diameter = 1 + 2 * RADIUS
    
    scaled_radius = RADIUS * GRID_SIZE
        
    start_lat = latitude - scaled_radius
    end_lat = latitude + scaled_radius
    start_lon = longitude - scaled_radius
    end_lon = longitude + scaled_radius
    
    param_folder = GRIB_FOLDER + grib_parameter + '/' + loc
    if not os.path.exists(param_folder):
        os.makedirs(param_folder)

    for year in years:
        
        year_folder = param_folder + '/' + str(year)
        if not os.path.exists(year_folder):
            os.makedirs(year_folder)
        
        f = open(GRIB_FOLDER + grib_parameter + '/' + str(year) + '.grib')
        month_it = 0
        month = 1
        day = 1
        hour = 0
        
        current_month = []
        current_day = np.zeros((4, diameter, diameter))
       
        while True:
            gc.collect()
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break

            dataDate = codes_get(gid, 'dataDate')
            
            dataDate_ = str(dataDate).zfill(8)
            month_ = int(dataDate_[4:6])
            day_ = int(dataDate_[6:8])
          
            if (month_ > month):
                current_month.append(current_day)
                month_array = np.asarray(current_month)
                print('added month ' + str(month))
                print('month shape:')
                print(month_array.shape)
                np.save(year_folder + '/' + str(month) + '.npy', month_array)
                month = month + 1
                day = 1
                hour = 0
                current_day = np.zeros((4, diameter, diameter))
                current_month = []
                
            elif (day_ > day):
                current_month.append(current_day)
                hour = 0
                day = day + 1
                current_day = np.zeros((4, diameter, diameter))
    
            bottomLeft = codes_grib_find_nearest(gid, start_lat, start_lon)[0]
            topRight = codes_grib_find_nearest(gid, end_lat, end_lon)[0]
            
            latIt = 0
            for lat in reversed(list(drange(bottomLeft.lat, topRight.lat + GRID_SIZE, GRID_SIZE))):
                lonIt = 0
                for lon in drange(bottomLeft.lon, topRight.lon + GRID_SIZE, GRID_SIZE):
                    nearest = codes_grib_find_nearest(gid, lat, lon)[0]
                    current_day[hour, latIt, lonIt] = nearest.value
                    lonIt = lonIt + 1  
                latIt = latIt + 1
        
            codes_release(gid)
            hour = hour + 1
            
        # save the last month
        current_month.append(current_day)
        month_array = np.asarray(current_month)
        print('added month ' + str(month))
        print('month shape:')
        print(month_array.shape)
        np.save(year_folder + '/' + str(month) + '.npy', month_array)
       
        f.close()

def save_location_uv(grib_parameter='u_ambient', binary_file='uv_ambient', binary_file_2='.bin_ambient20.bin', uv_param='u', loc='los_angeles', latitude=33.75, longitude=241.5):
    """ 
        Loads data from the binary files and saves it for further faster use
    """
    years = list(range(1979, 2017))
    months = list(range(1, 13))
    diameter = 1 + 2 * RADIUS
    
    scaled_radius = RADIUS * GRID_SIZE
        
    start_lat = int((latitude - scaled_radius) / GRID_SIZE)
    end_lat = int((latitude + scaled_radius) / GRID_SIZE)
    start_lon = int((longitude - scaled_radius) / GRID_SIZE)
    end_lon = int((longitude + scaled_radius) / GRID_SIZE)
    
    print(start_lat)
    print(end_lat)
    print(start_lon)
    print(end_lon)
    
    if (uv_param == 'u'):
        uv_param_index = 1
    else: 
        uv_param_index = 0

    param_folder = GRIB_FOLDER + grib_parameter + '/' + loc
    if not os.path.exists(param_folder):
        os.makedirs(param_folder)

    for year in years:
        gc.collect()
        if (calendar.isleap(year)):
            month_array = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            month_array = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        year_folder = param_folder + '/' + str(year)
        if not os.path.exists(year_folder):
            os.makedirs(year_folder)
        
        array = np.fromfile(GRIB_FOLDER + binary_file + '/' + str(year) + binary_file_2, dtype=np.float32)
        print(array.shape)
        
        array = np.reshape(array, (-1, 480, 2, 241))
        array = array[:, start_lon:end_lon+1, uv_param_index, start_lat:end_lat+1]
        print(array.shape)
      
        month = 0
        array_it = 0
       
        for month_a in month_array:
            month = month + 1
            gc.collect()
            current_month = np.zeros((month_a, 4, diameter, diameter))
            
            for day_it in range(month_a):
                current_month[day_it, :, :, :] = array[array_it:array_it + 4, :, :]
                array_it = array_it + 4

            print(current_month.shape)
            # swap to be in order day, time, lat, lon
            current_month = np.transpose(current_month, (0, 1, 3, 2))
            np.save(year_folder + '/' + str(month) + '.npy', current_month)

def main():
    #save_location('temperature')
    #save_location('wind_u')
    #save_location('wind_v') 
    #save_location('surface_pressure')
    #save_location('pressure') 
    #save_location('cloud_cover')
    save_location('total_precipitation', loc='zurich', latitude=47.25, longitude=8.25)
    save_location('total_precipitation', loc='paris', latitude=48.75, longitude=2.25)
    save_location('total_precipitation', loc='los_angeles', latitude=33.75, longitude=241.5)
    #save_location_uv('u_ambient', 'uv_ambient', '.bin_ambient20.bin', 'u')
    #save_location_uv('v_ambient', 'uv_ambient', '.bin_ambient20.bin', 'v')
    #save_location_uv('u_features', 'uv_features', '.bin_features20.bin', 'u')
    #save_location_uv('v_features', 'uv_features', '.bin_features20.bin', 'v')
    
    return 1

if __name__ == "__main__":
    sys.exit(main())
