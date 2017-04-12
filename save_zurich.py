import os, math, re, gc
import decimal
import numpy as np
import scipy as sp

from attrdict import AttrDict

from eccodes import *

GRID_SIZE = 0.75
RADIUS = 2
LATITUDE = 47.25
LONGITUDE = 8.25
GRIB_FOLDER = '/media/isa/VIS1/'

def drange(x, y, jump):
  x_ = decimal.Decimal(x)
  while x_ < y:
    yield float(x_)
    x_ += decimal.Decimal(jump)

def save_zurich(grib_parameter):
    """ 
        Loads data from the grib file and saves it for further faster use
    """
    years = list(range(1979, 2017))
    months = list(range(1, 13))
    diameter = 1 + 2 * RADIUS
    
    scaled_radius = RADIUS * GRID_SIZE
        
    start_lat = LATITUDE - scaled_radius
    end_lat = LATITUDE + scaled_radius
    start_lon = LONGITUDE - scaled_radius
    end_lon = LONGITUDE + scaled_radius
    
    param_folder = GRIB_FOLDER + grib_parameter + '/zurich'
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
                    current_day[hour, latIt, lonIt] = max(0, nearest.value)
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

def main():
    save_zurich('temperature')
    save_zurich('pressure')
    return 1

if __name__ == "__main__":
    sys.exit(main())