"""
    Loading and preprocessing of datasets
    Maybe augment?
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
  while x < y:
    yield float(x)
    x += decimal.Decimal(jump)
    
def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))

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
    
class DatasetSingle:
    """
        Loads data from a grid file and divides them into training and evaluation groups
        @param window_size: how many frames should be considered for the forecast
        @param grib_file: the path to the grib file to load
        @param start_lon: the start longitude to consider for the data cropping
        @param end_lon: the end longitude to consider for the data cropping
        @param start_lat: the start latitude to consider for the data cropping
        @param end_lat: the end latitude to consider for the data cropping
    """
    def __init__(self, params):
        self.check_params(params)
        self.load_frames(params)
        self.create_sets(params.window_size, params.test_fraction)
        
    def check_params(self, params):
        if params.test_fraction < 0 or params.test_fraction > 1:
            raise Exception("test fraction must be between 0 and 1")
            
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
         
        if params.max_frames < params.window_size * 2:
            raise Exception("max frames must be at least twice the window size")  
        
         
    def load_frames(self, params):
        """ Loads the training files """
        self.frames = []

        f = open(params.grib_file)
        
        index = 0
 
        while 1:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break

            nearest = codes_grib_find_nearest(gid, params.start_lat, params.start_lon)[0]
            #print params.start_lat, params.start_lon
            #print nearest.lat, nearest.lon, nearest.value, nearest.distance, \
            #nearest.index

            if nearest.value == codes_get_double(gid, "missingValue"):
                print ("missing")

            codes_release(gid)
            self.frames.append(max(0, nearest.value))
            
            index = index + 1
            print "loading frames: ", index, "\r",
            
            if index >= params.max_frames:
                break
        
        print ("")
        self.frames = np.expand_dims(np.asarray(self.frames), axis=1)
        f.close()
        
    def create_dataset(self, dataset, window_size):
        """ convert an array of values into a dataset matrix """
        dataX, dataY = [], []
        for i in range(len(dataset) - window_size - 1):
            a = dataset[i:(i + window_size), 0]
            dataX.append(a)
            dataY.append(dataset[i + window_size, 0])
        return np.array(dataX), np.array(dataY)
    
    def create_sets(self, window_size, test_fraction):
        """ split data into training/test set and shape/format. """
        # normalize the dataset
        self.frames = self.frames.astype('float32')
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.frames = self.scaler.fit_transform(self.frames)
        
        # split into train and test sets
        train_size = int(len(self.frames) * (1 - test_fraction))
        test_size = len(self.frames) - train_size
        print("loading %i training frames..." % train_size)
        print("loading %i testing frames..." % test_size)
        train, test = self.frames[0:train_size,:], self.frames[train_size:len(self.frames),:]

        # reshape into X=t and Y=t+1
        self.trainX, self.trainY = self.create_dataset(train, window_size)
        self.testX, self.testY = self.create_dataset(test, window_size)
        
        # reshape input to be [samples, time steps, features]
        self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
        self.testX = np.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))
        
        print ("shape of train data X is:")
        print (self.trainX.shape)
        print ("shape of train data Y is:")
        print (self.trainY.shape)
        print ("shape of test data X is:")
        print (self.testX.shape)
        print ("shape of test data Y is:")
        print (self.testY.shape)

    
class DatasetMultiple:
    """
        Loads data from a grid file and divides them into training and evaluation groups
        @param window_size: how many frames should be considered for the forecast
        @param grib_file: the path to the grib file to load
        @param start_lon: the start longitude to consider for the data cropping
        @param end_lon: the end longitude to consider for the data cropping
        @param start_lat: the start latitude to consider for the data cropping
        @param end_lat: the end latitude to consider for the data cropping
    """
    def __init__(self, params):
        self.check_params(params)
        self.load_frames(params)
        self.create_sets(params.window_size, params.test_fraction)
        
    def check_params(self, params):
        if params.test_fraction < 0 or params.test_fraction > 1:
            raise Exception("test fraction must be between 0 and 1")
            
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
         
        if params.max_frames < params.window_size * 2:
            raise Exception("max frames must be at least twice the window size")  
            
        self.vector_size = int(1 + ((round_nearest(params.end_lat, GRID_SIZE) - round_nearest(params.start_lat, GRID_SIZE)) / GRID_SIZE) * \
            1 + ((round_nearest(params.end_lon, GRID_SIZE) - round_nearest(params.start_lon, GRID_SIZE)) / GRID_SIZE))
            
        print ("vector size is %i" % self.vector_size)
        
         
    def load_frames(self, params):
        """ Loads the training files """
        self.frames = []

        f = open(params.grib_file)
        
        index = 0
 
        while 1:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
                
            frame = np.empty([self.vector_size])
            frameIt = 0
            
            topLeft = codes_grib_find_nearest(gid, params.start_lat, params.start_lon)[0]
            bottomRight = codes_grib_find_nearest(gid, params.end_lat, params.end_lon)[0]
            
            for lat in drange(topLeft.lat, bottomRight.lat, 0.75):
                for lon in drange(topLeft.lon, bottomRight.lon, 0.75):
                    nearest = codes_grib_find_nearest(gid, lat, lon)[0]
                    frame[frame_it] = max(0, nearest.value)
                    if nearest.value == codes_get_double(gid, "missingValue"):
                        print ("missing")
                    frameIt = frameIt + 1
                    
            #print params.start_lat, params.start_lon
            #print nearest.lat, nearest.lon, nearest.value, nearest.distance, \
            #nearest.index

            codes_release(gid)
            self.frames.append(frame)
            
            index = index + 1
            print "loading frames: ", index, "\r",
            
            if index >= params.max_frames:
                break
        
        print ("")
        self.frames = np.asarray(self.frames)
        print ("frames shape")
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
    
    def create_sets(self, window_size, test_fraction):
        """ split data into training/test set and shape/format. """
        # normalize the dataset
        self.frames = self.frames.astype('float32')
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.frames = self.scaler.fit_transform(self.frames)
        
        # split into train and test sets
        train_size = int(len(self.frames) * (1 - test_fraction))
        test_size = len(self.frames) - train_size
        print("loading %i training frames..." % train_size)
        print("loading %i testing frames..." % test_size)
        train, test = self.frames[0:train_size,:], self.frames[train_size:len(self.frames),:]

        # reshape into X=t and Y=t+1
        self.trainX, self.trainY = self.create_dataset(train, window_size)
        self.testX, self.testY = self.create_dataset(test, window_size)
        
        # reshape input to be [samples, time steps, features]
        #self.trainX = np.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
        #self.testX = np.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))
        
        #self.trainX = np.random.rand(self.trainX.shape[0], 2, self.trainX.shape[1])
        #self.testX = np.random.rand(self.testX.shape[0], 2, self.testX.shape[1])
        #self.trainY = np.random.rand(self.trainY.shape[0], 2)
        #self.testY = np.random.rand(self.testY.shape[0], 2)

        print ("shape of train data X is:")
        print (self.trainX.shape)
        print ("shape of train data Y is:")
        print (self.trainY.shape)
        print ("shape of test data X is:")
        print (self.testX.shape)
        print ("shape of test data Y is:")
        print (self.testY.shape)

