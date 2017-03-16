"""
    Loading and preprocessing of datasets
    from grib files
"""
import os, math, re
import decimal
import numpy as np
import scipy as sp
from sklearn.preprocessing import MinMaxScaler
from attrdict import AttrDict

from eccodes import *

GRID_SIZE = 0.75

def drange(x, y, jump):
  x_ = decimal.Decimal(x)
  while x_ < y:
    yield float(x_)
    x_ += decimal.Decimal(jump)
    
def round_nearest(x, a):
    return round(x / a) * a

def crop_center(img, width, height):
    """
        returns the center of the image
        @param width: width of cropped image
        @param height: height of cropped image
    """
    y,x,z = img.shape
    startx = x//2-(width//2)
    starty = y//2-(height//2)    
    return img[starty:starty + height, startx:startx + width, :]
    
def lonlat_to_idx(lon, lat, grid_lon=GRID_SIZE, grid_lat=GRID_SIZE):
    """ 
        Converts longitude and latitude to the corresponding 
        indexes in a data matrix, given the grid density
    """
    return int((-lat + 90) / grid_lat), int(lon / grid_lon)
    
class FrameParameters:
    """
        Parameters belonging to a training or test frame.
        @param date: the date (string/int) from where the frame is from
        @param time: the time (string/int) where the data is from
        @param is_new: whether this is the start of a new sequence
    """
    def __init__(self, date, time, is_new):
        self.date = str(date).zfill(8)
        self.time = str(time).zfill(4)
        self.is_new = is_new
    
class Dataset:
    """
        Superclass for loading dataset from grib files
    """
    def __init__(self, params):
        self.check_params(params)
        self.load_frames()
        self.normalize_frames()
        self.create_samples()
        
    def include_date(self, dataDate):
        """ 
            Returns whether dataDate should be included
            with respect years and months given in the params
            @param dataDate: date as a string/int
        """
        dataDate_ = str(dataDate).zfill(8)
        year = int(dataDate_[:4])
        month = int(dataDate_[4:6])

        return (year in self.params.years and month in self.params.months)
        
    def check_params(self, params):
        if not hasattr(params, 'max_frames'):
            params.max_frames = float('inf')
        
        if params.max_frames < params.window_size + 1:
            raise Exception("max frames must be at least window_size + 1")
        
        if params.start_lat > params.end_lat:
            raise Exception("latitude dimensions do not match")
            
        if params.start_lon > params.end_lon:
            raise Exception("longitude dimensions do not match")  

        if params.start_lon > 360 or params.start_lon < 0:
            raise Exception("longitude (start) must be between 0 and 360")   
            
        if params.end_lon > 360 or params.end_lon < 0:
            raise Exception("longitude (end) must be between 0 and 360")  
            
        if params.start_lat > 90 or params.start_lat < -90:
            raise Exception("latitude (start) must be between -90 and 90")   
            
        if params.end_lat > 90 or params.end_lat < -90:
            raise Exception("latitude (end) must be between -90 and 90") 
            
        if params.window_size < 1:
            raise Exception("window size must be at least one") 

        nelat = round_nearest(params.end_lat, GRID_SIZE)
        nslat = round_nearest(params.start_lat, GRID_SIZE)
        nelon = round_nearest(params.end_lon, GRID_SIZE)
        nslon = round_nearest(params.start_lon, GRID_SIZE)

        self.lat_range = int((1 + (nelat - nslat) / GRID_SIZE))
        self.lon_range = int((1 + (nelon - nslon) / GRID_SIZE))
        self.vector_size = self.lat_range * self.lon_range
            
        print ("vector size is %i" % self.vector_size)
        
        self.params = params
        
    def normalize_frames(self):
        """ normalizes loaded frames """
        raise NotImplementedError()
         
    def load_frames(self):
        """ Loads data from the grib file """
        self.frames = []
        self.frames_data = []

        f = open(self.params.grib_file)
        
        index = 0
        is_new = True
 
        while 1:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
                
            # check if this matches our month and year
            dataDate = codes_get(gid, "dataDate")
            if not self.include_date(dataDate):
                is_new = True
                print "skipping date: " + str(dataDate).zfill(8), "            \r",
                codes_release(gid)
                continue
                
            frame = np.empty([self.vector_size])
            frameIt = 0
            
            bottomLeft = codes_grib_find_nearest(gid, self.params.start_lat, self.params.start_lon)[0]
            topRight = codes_grib_find_nearest(gid, self.params.end_lat, self.params.end_lon)[0]
            
            for lat in reversed(list(drange(bottomLeft.lat, topRight.lat + GRID_SIZE, GRID_SIZE))):
                for lon in drange(bottomLeft.lon, topRight.lon + GRID_SIZE, GRID_SIZE):
                    nearest = codes_grib_find_nearest(gid, lat, lon)[0]
                    frame[frameIt] = max(0, nearest.value)
                    if nearest.value == codes_get_double(gid, "missingValue"):
                        raise Warning("missing value!")
                    frameIt = frameIt + 1
            
            self.frames.append(frame)
            self.frames_data.append(FrameParameters(dataDate, codes_get(gid, "dataTime"), \
            is_new))
            is_new = False 
           
            codes_release(gid)
            index = index + 1
            print "loading frames: ", index, "                 \r",
            
            if index > self.params.max_frames:
                break
    
        print ("")
        self.frames = np.asarray(self.frames)
        print ("frames shape:")
        print (self.frames.shape)
        f.close()

    def create_dataset(self, dataset, window_size):
        """ convert an array of values into a dataset matrix """
        dataX, dataY = [], []
        for i in range(len(dataset) - window_size - 1):
            a = dataset[i:(i + window_size), :]
            dataX.append(a)
            dataY.append(dataset[i + window_size, :])
        return np.array(dataX), np.array(dataY)
    
    def create_samples(self):
        """ split data into x and y parts and shapes them. """
        # used later to map the frames to their frame parameters
        self.frames_idx = []
        frames_idx = 0

        # split into train and test sets
        frame_sets = []
        frames = []

        for index in range(len(self.frames)):
            if (len(frames) > 0 and self.frames_data[index].is_new):
                frame_sets.append(np.asarray(frames))
                frames = []
                self.frames_idx = self.frames_idx[:-self.params.window_size or None]
            
            frames.append(self.frames[index])
            self.frames_idx.append(frames_idx)
            frames_idx = frames_idx + 1
                
        if (len(frames) > 0):
            frame_sets.append(np.asarray(frames))
            frames = []
            self.frames_idx = self.frames_idx[:-self.params.window_size or None]
        
        self.dataX = np.array([])
        self.dataY = np.array([])

        for frame_set in frame_sets:
            dX, dY = self.create_dataset(frame_set, self.params.window_size)
            self.dataX = np.concatenate((self.dataX, dX), 0) if self.dataX.size else dX
            self.dataY = np.concatenate((self.dataY, dY), 0) if self.dataY.size else dY
      
        print ("shape of data X is:")
        print (self.dataX.shape)
        print ("shape of data Y is:")
        print (self.dataY.shape)

class DatasetTrain(Dataset):
    """
        Loads training data from a grid file, shapes and formats it.
        @param params.window_size: how many frames should be considered for the forecast
        @param params.max_frames: the maximum frames to load
        @param params.grib_file: the path to the grib file to load
        @param params.start_lon: the start longitude to consider for the data cropping
        @param params.end_lon: the end longitude to consider for the data cropping
        @param params.start_lat: the start latitude to consider for the data cropping
        @param params.end_lat: the end latitude to consider for the data cropping
        @param params.years: only load data from those years
        @param params.months: only load data from those months
    """
    def __init__(self, params):
        Dataset.__init__(self, params)

    def normalize_frames(self):
        """ normalizes loaded frames """
        self.frames = self.frames.astype('float32')
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.frames = self.scaler.fit_transform(self.frames)
        
    def get_scaler_params(self):
        """ 
            returns the params of the MinMaxScaler
            which can then be used to create a scaler for the test set
        """
        return self.scaler.get_params(deep=True)
        
class DatasetTest(Dataset):
    """
        Loads test data from a grid file, shapes and formats it.
        @param params.window_size: how many frames should be considered for the forecast
        @param params.max_frames: the maximum frames to load
        @param params.grib_file: the path to the grib file to load
        @param params.start_lon: the start longitude to consider for the data cropping
        @param params.end_lon: the end longitude to consider for the data cropping
        @param params.start_lat: the start latitude to consider for the data cropping
        @param params.end_lat: the end latitude to consider for the data cropping
        @param params.years: only load data from those years
        @param params.months: only load data from those months
        @param scaler_params: scaler params gotten from the training set
    """
    def __init__(self, params, scaler_params):
        print(scaler_params)
        self.scaler = MinMaxScaler()
        self.scaler.set_params(**scaler_params)
        Dataset.__init__(self, params)

    def normalize_frames(self):
        """ normalizes loaded frames """
        self.frames = self.frames.astype('float32')
        self.frames = self.scaler.transform(self.frames)
