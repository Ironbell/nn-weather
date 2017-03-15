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
        Parameters belonging to a training and/or test frame.
        @param date: the date (string/int) from where the frame is from
        @param time: the time (string/int) where the data is from
        @param is_train: whether this belongs to the training data
        @param is_test: this belongs to the testdata
        @param is_new_train: whether this is the start of a new train sequence
        @param is_new_test: whether this is the start of a new test sequence
    """
    def __init__(self, date, time, is_train, is_test, is_new_train, is_new_test):
        self.date = str(date)
        self.time = str(time)
        self.is_train = is_train
        self.is_test = is_test
        self.is_new_train = is_new_train
        self.is_new_test = is_new_test
    
class Dataset:
    """
        Superclass for loading dataset from grib files
    """
    def __init__(self, params):
        self.check_params(params)
        self.load_frames()
        self.create_sets()
        
    def include_date(self, dataDate):
        """ 
            Returns whether dataDate should be included
            in train and/or test data
            with respect to train and test dates given in the params
            @param dataDate: date as a string
        """
        dataDate_ = str(dataDate)
        year = int(dataDate_[:4])
        month = int(dataDate_[4:6])

        is_train = (year in self.params.train_years and month in self.params.train_months)
        is_test = (year in self.params.test_years and month in self.params.test_months)

        return is_train, is_test
        
    def check_params(self, params):
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
         
    def load_frames(self):
        """ Loads the training files """
        raise NotImplementedError()

    def create_dataset(self, dataset, window_size):
        """ convert an array of values into a dataset matrix """
        raise NotImplementedError()
    
    def create_sets(self):
        """ split data into training/test set and shape/format. """
        raise NotImplementedError()

class DatasetMultiple(Dataset):
    """
        Loads data from a grid file and divides them into training and evaluation groups
        @param window_size: how many frames should be considered for the forecast
        @param grib_file: the path to the grib file to load
        @param start_lon: the start longitude to consider for the data cropping
        @param end_lon: the end longitude to consider for the data cropping
        @param start_lat: the start latitude to consider for the data cropping
        @param end_lat: the end latitude to consider for the data cropping
        @param train_years: only load train data from those years
        @param test_years: only load test data from those years
        @param train_months: only load train data from those months
        @param test_months: only load test data from those months
    """
    def __init__(self, params):
        Dataset.__init__(self, params)
        
    def load_frames(self):
        """ Loads the training files """
        self.frames = []
        self.frames_data = []

        f = open(self.params.grib_file)
        
        index = 0
        is_new_train = True
        is_new_test = True
 
        while 1:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
                
            # check if this matches our month and year for train and/or test
            is_train, is_test = self.include_date(codes_get(gid, "dataDate"))
            
            if (not is_train and not is_test):
                is_new_train = True
                is_new_test = True
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
            self.frames_data.append(FrameParameters(codes_get(gid, "dataDate"), codes_get(gid, "dataTime"), \
            is_train, is_test, is_new_train, is_new_test))
            is_new_train = not is_train 
            is_new_test = not is_test
            
            codes_release(gid)
            index = index + 1
            print "loading frames: ", index, "\r",
            
            if index > 1000:
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
    
    def create_sets(self):
        """ split data into training/test set and shape/format. """
        # normalize the dataset
        self.frames = self.frames.astype('float32')
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.frames = self.scaler.fit_transform(self.frames)
        
        # split into train and test sets
        train_frames = []
        train = []
        test_frames = []
        test = []
        
        self.test_frame_idx = []
        self.train_frame_idx = []
 
        for index in range(len(self.frames)):
            if (self.frames_data[index].is_train):
                if (len(train) > 0 and self.frames_data[index].is_new_train):
                    train_frames.append(np.asarray(train))
                    train = []
                
                train.append(self.frames[index])
                self.train_frame_idx.append(index)
                
            if (self.frames_data[index].is_test):
                if (len(test) > 0 and self.frames_data[index].is_new_test):
                    test_frames.append(np.asarray(test))
                    test = []
                
                test.append(self.frames[index])
                self.test_frame_idx.append(index)
                
        if (len(train) > 0):
            train_frames.append(np.asarray(train))
            train = []
        
        if (len(test) > 0):
            test_frames.append(np.asarray(test))
            test = []

        self.trainX = np.array([])
        self.trainY = np.array([])
        self.testX = np.array([])
        self.testY = np.array([])
        
        for train in train_frames:
            tX, tY = self.create_dataset(train, self.params.window_size)
            self.trainX = np.concatenate((self.trainX, tX), 0) if self.trainX.size else tX
            self.trainY = np.concatenate((self.trainY, tY), 0) if self.trainY.size else tY
        
        for test in test_frames:
            tX, tY = self.create_dataset(test, self.params.window_size)
            self.testX = np.concatenate((self.testX, tX), 0) if self.testX.size else tX
            self.testY = np.concatenate((self.testY, tY), 0) if self.testY.size else tY

        print ("shape of train data X is:")
        print (self.trainX.shape)
        print ("shape of train data Y is:")
        print (self.trainY.shape)
        print ("shape of test data X is:")
        print (self.testX.shape)
        print ("shape of test data Y is:")
        print (self.testY.shape)

