"""
    Loading and preprocessing of datasets
    from grib files
"""
import os, math, re, gc
import decimal
import numpy as np
import scipy as sp

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
        
    def include_datetime(self, dataDate, dataTime):
        """ 
            Returns whether dataDate and dataTime should be included
            with to the months given in the params
            @param dataDate: date as a string/int
            @param dataTime: time as a string/int
            @return includeDate, includeTime (boolean)
        """
        dataDate_ = str(dataDate).zfill(8)
        dataTime_ = str(dataTime).zfill(4)
        month = int(dataDate_[4:6])
        hour = int(dataTime_[0:2])

        return (month in self.params.months), (hour in self.params.hours)
        
    def check_params(self, params):
        """ 
            Checks the parameters for validity
        """
        if not hasattr(params, 'forecast_distance'):
            params.forecast_distance = 0
            
        if params.forecast_distance < 0:
            raise Exception("forecast distance must be at least 0")
            
        if not hasattr(params, 'steps_before'):
            params.steps_before = 1
            
        if params.steps_before < 1:
            raise Exception("steps before must be at least 1")
            
        if not hasattr(params, 'steps_after'):
            params.steps_after = 1
            
        if params.steps_after < 1:
            raise Exception("steps ahead must be at least 1")
            
        if not hasattr(params, 'max_frames'):
            params.max_frames = float('inf')
        
        if params.max_frames < params.steps_before + params.steps_after + params.forecast_distance:
            raise Exception("max frames must be at least steps_before + steps_after + forecast_distance")
            
        if len(params.grib_parameters) < 1:
            raise Exception("grib_parameters must be a list with at least one entry")
        
    def normalize_frames(self):
        """ 
            Normalizes and reshapes loaded frames 
        """
        self.frames = self.frames.astype('float32')
        
        nb_params = len(self.params.grib_parameters)
        self.scalers = np.empty((nb_params, 2))
        
        # we normalize per grib  parameter
        for i in range(nb_params):
            array = self.frames[:,:,i]
            min_data, ptp_data = array.min(), array.ptp()
            array = (array - min_data) / ptp_data
            self.scalers[i, 0] = min_data
            self.scalers[i, 1] = ptp_data
            self.frames[:,:,i] = array
            
        self.frames = self.frames.reshape(self.frames.shape[0], -1)

    def inverse_transform_data(self, flatten=True):
        """
            @return unscaled (true) dataX and dataY
        """     
        nb_params = len(self.params.grib_parameters)
        dataX = np.copy(self.dataX.reshape(self.dataX.shape[0], self.dataX.shape[1], self.params.nb_grib_points, nb_params))

        for i in range(dataX.shape[3]):
            dataX[:,:,:,i] = dataX[:,:,:,i] * self.scalers[i, 1] + self.scalers[i, 0]

        if (flatten):
            dataX = dataX.reshape(self.dataX.shape) 
        
        dataY = np.copy(self.dataY.reshape(self.dataY.shape[0], self.dataY.shape[1], self.params.nb_grib_points, nb_params))
        for i in range(dataY.shape[3]):
            dataY[:,:,:,i] = dataY[:,:,:,i] * self.scalers[i, 1] + self.scalers[i, 0]
        
        if (flatten):
            dataY = dataY.reshape(self.dataY.shape) 
      
        return dataX, dataY
  
    def predict_data(self, model, flatten=True):
        """
            predicts dataY with the given model
            using dataX as input and unscales it
            @param flatten: whether to flatten the reshaped data or leave it seperated per grib parameter
            @return unscaled prediction of shape (nb_samples, steps_after, features)
        """
        
        predict = model.predict(self.dataX)
        
        nb_params = len(self.params.grib_parameters)
        predict = predict.reshape(predict.shape[0], predict.shape[1], self.params.nb_grib_points, nb_params)
        
        for i in range(predict.shape[3]):
            predict[:,:,:,i] = predict[:,:,:,i] * self.scalers[i, 1] + self.scalers[i, 0]
          
        if (flatten):
            predict = predict.reshape(self.dataY.shape) 
            
        return predict
        
    def predict_constant(self, flatten=True):
        """
            predicts dataY as constant last element from dataX and unscales it
            @param flatten: whether to flatten the reshaped data or leave it seperated per grib parameter
            @return unscaled prediction of shape (nb_samples, steps_after, features)
        """
        
        predict = np.empty(self.dataY.shape)
        for i in range(predict.shape[1]):
            predict[:,i,:] = self.dataX[:,-1,:]
        
        nb_params = len(self.params.grib_parameters)
        predict = predict.reshape(predict.shape[0], predict.shape[1], self.params.nb_grib_points, nb_params)
        
        for i in range(predict.shape[3]):
            predict[:,:,:,i] = predict[:,:,:,i] * self.scalers[i, 1] + self.scalers[i, 0]
          
        if (flatten):
            predict = predict.reshape(self.dataY.shape) 
            
        return predict

    def load_frames(self):
        """ 
            Loads data from the grib file 
            Implemented in subclasses
        """
        raise NotImplementedError()

    def create_dataset(self, dataset):
        """ 
            convert an array of values into a dataset matrix 
        """
        steps_before = self.params.steps_before
        steps_after = self.params.steps_after
        forecast_distance = self.params.forecast_distance

        dataX, dataY = [], []
        for i in range(len(dataset) - steps_before - forecast_distance - steps_after - 1):
            a = dataset[i:(i + steps_before), :]
            dataX.append(a)
            dataY.append(dataset[(i + steps_before + forecast_distance):(i + steps_before + forecast_distance + steps_after), :])
        return np.array(dataX), np.array(dataY)
    
    def create_samples(self):
        """ 
            split data into x and y parts and shapes them. 
        """
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
                self.frames_idx = self.frames_idx[:-self.params.steps_before or None]
            
            frames.append(self.frames[index])
            self.frames_idx.append(frames_idx)
            frames_idx = frames_idx + 1
                
        if (len(frames) > 0):
            frame_sets.append(np.asarray(frames))
            frames = []
        
        self.dataX = np.array([])
        self.dataY = np.array([])

        for frame_set in frame_sets:
            dX, dY = self.create_dataset(frame_set)
            self.dataX = np.concatenate((self.dataX, dX), 0) if self.dataX.size else dX
            self.dataY = np.concatenate((self.dataY, dY), 0) if self.dataY.size else dY
      
        print ("shape of data X is:")
        print (self.dataX.shape)
        print ("shape of data Y is:")
        print (self.dataY.shape)
        
        print('shape of frames idx is:')
        print(len(self.frames_idx))
        
        print('length of frames data is:')
        print(len(self.frames_data))

class DatasetArea(Dataset):
    """
        Loads training/test data in a certain lon/lat range from a grid file, shapes and formats it.
        @param params.steps_before: how many frames before does the network take as input (1 default)
        @param params.steps_after: how many frames ahead it should predict. (1 default)
        @param params.forecast_distance: how many frames it should skip when predicting (0 default)
        @param params.max_frames: the maximum frames to load
        @param params.grib_folder: the path to the folder holding grib subfolders
        @param params.grib_parameters: which parameters to consider, for example ['temperature', 'pressure']
        @param params.start_lon: the start longitude to consider for the data cropping
        @param params.end_lon: the end longitude to consider for the data cropping
        @param params.start_lat: the start latitude to consider for the data cropping
        @param params.end_lat: the end latitude to consider for the data cropping
        @param params.years: only load data from those years
        @param params.months: only load data from those months
        @param params.hours: only load data from those hours
    """
    def __init__(self, params):
        Dataset.__init__(self, params)

    def check_params(self, params):
        """ 
            Checks the parameters for validity
        """
        Dataset.check_params(self, params)
        
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

        nelat = round_nearest(params.end_lat, GRID_SIZE)
        nslat = round_nearest(params.start_lat, GRID_SIZE)
        nelon = round_nearest(params.end_lon, GRID_SIZE)
        nslon = round_nearest(params.start_lon, GRID_SIZE)

        self.lat_range = int((1 + (nelat - nslat) / GRID_SIZE))
        self.lon_range = int((1 + (nelon - nslon) / GRID_SIZE))

        params.nb_grib_points = self.lat_range * self.lon_range
        params.nb_features = params.nb_grib_points * len(params.grib_parameters)
            
        print ('nb features is %i' % params.nb_features)
        print ('nb grib points is %i' % params.nb_grib_points)
        print ('parameters are: ')
        print (params.grib_parameters)

        self.params = params

    def load_frames(self):
        """ 
            Loads data from the grib file 
        """
        self.frames = []
        self.frames_data = []
        
        index = 0
        is_new = True

        for year in self.params.years:
            files = []
            for parameter in self.params.grib_parameters:
                f = open(self.params.grib_folder + parameter + "/" + str(year) + '.grib')
                files.append(f)
                
            while index <= self.params.max_frames:
                gc.collect()
                gids = []
                for f in files:
                    gid = codes_grib_new_from_file(f)
                    gids.append(gid)
                if gids[0] is None:
                    break
                    
                missingVal = codes_get_double(gids[0], 'missingValue')
                    
                # check if this matches our month, year and hour
                dataDate = codes_get(gids[0], 'dataDate')
                dataTime = codes_get(gids[0], 'dataTime')
                include_date, include_time = self.include_datetime(dataDate, dataTime)
                if not (include_date and include_time):
                    is_new = not include_date
                    print 'skipping date: ' + str(dataDate).zfill(8), '            \r',
                    for gid in gids:
                        codes_release(gid)
                    continue
                    
                frame = np.empty((self.params.nb_grib_points, len(self.params.grib_parameters)))  
                gidIt = 0
                for gid in gids:
                    frameIt = 0
                    
                    bottomLeft = codes_grib_find_nearest(gid, self.params.start_lat, self.params.start_lon)[0]
                    topRight = codes_grib_find_nearest(gid, self.params.end_lat, self.params.end_lon)[0]
                    
                    for lat in reversed(list(drange(bottomLeft.lat, topRight.lat + GRID_SIZE, GRID_SIZE))):
                        for lon in drange(bottomLeft.lon, topRight.lon + GRID_SIZE, GRID_SIZE):
                            nearest = codes_grib_find_nearest(gid, lat, lon)[0]
                            frame[frameIt, gidIt] = max(0, nearest.value)
                            if nearest.value == missingVal:
                                raise Warning('missing value!')
                            frameIt = frameIt + 1
                            
                    gidIt = gidIt + 1
                
                self.frames.append(frame)
                self.frames_data.append(FrameParameters(dataDate, dataTime, is_new))
                is_new = False 
               
                for gid in gids:
                    codes_release(gid)
                index = index + 1
                print 'loading frames: ', index, '                 \r',

            for f in files:
                f.close()
            if index > self.params.max_frames:
                break
            
        print ('')
        self.frames = np.asarray(self.frames)
        gc.collect()
        print ('frames shape:')
        print (self.frames.shape)
        
class DatasetNearest(Dataset):
    """
        Loads training/test data of the nearest n points from a grid file, shapes and formats it.
        @param params.steps_before: how many frames before does the network take as input (1 default)
        @param params.steps_after: how many frames ahead it should predict. (1 default)
        @param params.forecast_distance: how many frames it should skip when predicting (0 default)
        @param params.max_frames: the maximum frames to load
        @param params.grib_folder: the path to the folder holding grib subfolders
        @param params.grib_parameters: which parameters to consider, for example ['temperature', 'pressure']
        @param params.lon: the longitude of the center point
        @param params.lat: the latitude of the center point
        @param params.npoints: how many points to load
        @param params.years: only load data from those years
        @param params.months: only load data from those months
        @param params.hours: only load data from those hours
    """
    def __init__(self, params):
        Dataset.__init__(self, params)
        
    def check_params(self, params):
        """ 
            Checks the parameters for validity
        """
        Dataset.check_params(self, params)

        if params.lon > 360 or params.lon < 0:
            raise Exception('longitude must be between 0 and 360')   

        if params.lat > 90 or params.lat < -90:
            raise Exception('latitude must be between -90 and 90')   

        if params.npoints < 1:
            raise Exception('n points must be at least one') 

        params.lat = round_nearest(params.lat, GRID_SIZE)
        params.lon = round_nearest(params.lon, GRID_SIZE)
       
        params.nb_features = params.npoints * len(params.grib_parameters)
        params.nb_grib_points = params.npoints
            
        print ('nb features is %i' % params.nb_features)
        print ('nb grib points is %i' % params.nb_grib_points)
        print ('parameters are: ')
        print (params.grib_parameters)
        self.params = params
        
    def calculate_npoints(self):
        """ 
            returns a list of the nearest n points (lat, lon)
        """
        nnearest = []
        
        bound_range = int(math.ceil(math.sqrt(self.params.npoints)))
        if bound_range % 2 == 0:
            bound_range = bound_range // 2
        else:
            bound_range = (bound_range - 1) // 2
            
        for x in range(-bound_range, bound_range + 1):
            for y in range(-bound_range, bound_range + 1):
                nnearest.append((x, y))
                
        nnearest.sort(key=lambda tup: (tup[0] * tup[0] + tup[1] * tup[1]))

        nnearest = nnearest[:self.params.npoints]
        nnearest = [tuple((GRID_SIZE * tup[0] + self.params.lat, GRID_SIZE * tup[1] + self.params.lon)) for tup in nnearest] 

        return nnearest

    def load_frames(self):
        """ Loads data from the grib file """
        self.frames = []
        self.frames_data = []
        
        index = 0
        is_new = True
        nnearest = self.calculate_npoints()

        for year in self.params.years:
            files = []
            for parameter in self.params.grib_parameters:
                f = open(self.params.grib_folder + parameter + "/" + str(year) + '.grib')
                files.append(f)
                
            while index <= self.params.max_frames:
                gc.collect()
                gids = []
                for f in files:
                    gid = codes_grib_new_from_file(f)
                    gids.append(gid)
                if gids[0] is None:
                    break
                    
                missingVal = codes_get_double(gids[0], 'missingValue')
                    
                # check if this matches our month, year and hour
                dataDate = codes_get(gids[0], 'dataDate')
                dataTime = codes_get(gids[0], 'dataTime')
                include_date, include_time = self.include_datetime(dataDate, dataTime)
                if not (include_date and include_time):
                    is_new = not include_date
                    print 'skipping date: ' + str(dataDate).zfill(8), '            \r',
                    for gid in gids:
                        codes_release(gid)
                    continue
                    
                frame = np.empty((self.params.nb_grib_points, len(self.params.grib_parameters)))  
                gidIt = 0
                
                for gid in gids:
                    frameIt = 0
                    for nearest_point in nnearest:
                        nearest = codes_grib_find_nearest(gid, nearest_point[0], nearest_point[1])[0]
                        frame[frameIt, gidIt] = max(0, nearest.value)
                        if nearest.value == missingVal:
                            raise Warning('missing value!')
                        frameIt = frameIt + 1
                        
                    gidIt = gidIt + 1
            
                self.frames.append(frame)
                self.frames_data.append(FrameParameters(dataDate, dataTime, is_new))
                is_new = False 
               
                for gid in gids:
                    codes_release(gid)
                index = index + 1
                print 'loading frames: ', index, '                 \r',

            for f in files:
                f.close()
            if index > self.params.max_frames:
                break
            
        print ('')
        self.frames = np.asarray(self.frames)
        gc.collect()
        print ('frames shape:')
        print (self.frames.shape)
        
class DatasetSquareArea(Dataset):
    """
        Loads training/test data of a square area around the center from a grid file, shapes and formats it.
        @param params.steps_before: how many frames before does the network take as input (1 default)
        @param params.forecast_distance: how many frames it should skip when predicting (0 default)
        @param params.max_frames: the maximum frames to load
        @param params.grib_folder: the path to the folder holding grib subfolders
        @param params.grib_parameters: which parameters to consider, for example ['temperature', 'pressure']
        @param params.lon: the longitude of the center point
        @param params.lat: the latitude of the center point
        @param params.radius: the radius around the center
        @param params.years: only load data from those years
        @param params.months: only load data from those months
        @param params.hours: only load data from those hours
    """
    def __init__(self, params):
        params.steps_after = 1
        Dataset.__init__(self, params)
        
    def check_params(self, params):
        """ 
            Checks the parameters for validity
        """
        Dataset.check_params(self, params)
        
        if not hasattr(params, 'is_zurich'):
            params.is_zurich = False

        if params.lon > 360 or params.lon < 0:
            raise Exception('longitude must be between 0 and 360')   

        if params.lat > 90 or params.lat < -90:
            raise Exception('latitude must be between -90 and 90')   

        if params.radius < 0 or params.radius > 10:
            raise Exception('radius has to be between 0 and 10') 

        params.lat = round_nearest(params.lat, GRID_SIZE)
        params.lon = round_nearest(params.lon, GRID_SIZE)
        
        scaled_radius = params.radius * GRID_SIZE
        
        params.start_lat = params.lat - scaled_radius
        params.end_lat = params.lat + scaled_radius
        params.start_lon = params.lon - scaled_radius
        params.end_lon = params.lon + scaled_radius

        params.nb_grib_points = (1 + 2 * params.radius) * (1 + 2 * params.radius)
        params.nb_features = params.nb_grib_points * len(params.grib_parameters)

        print ('nb features is %i' % params.nb_features)
        print ('nb grib points is %i' % params.nb_grib_points)
        print ('parameters are: ')
        print (params.grib_parameters)
        self.params = params
        
    def load_frames(self):
        """ 
            Loads data from the grib file or directly from the np file
        """
        if (self.params.is_zurich):
            self.load_frames_zurich()
        else:
            self.load_frames_default()
    
    def load_frames_zurich(self):
        self.frames_data = []
        self.frames = []

        include_hour = [0 in self.params.hours, 6 in self.params.hours, 12 in self.params.hours, 18 in self.params.hours]
        nb_params = len(self.params.grib_parameters)
        diameter = 1 + 2 * self.params.radius

        old_month = -1
        for year in self.params.years:
            for month in self.params.months:
                is_new = (old_month % 12) + 1 != month
                old_month = month
                ym_arrays = []
                gc.collect()
                
                for parameter in self.params.grib_parameters:
                    ym_array = np.load(self.params.grib_folder + parameter + '/zurich/' + str(year) + '/' + str(month) + '.npy')
                    ym_arrays.append(ym_array)

                for day in range(ym_arrays[0].shape[0]):
                    for hour in range(ym_arrays[0].shape[1]):
                        if not include_hour[hour]:
                            continue
                        
                        frame = np.empty((diameter, diameter, nb_params))  
                        for p in range(nb_params):
                            frame[:, :, p] = ym_arrays[p][day, hour, :, :]
                        
                        self.frames.append(frame)
                        dataDate = str(year + 1979) + str(month + 1)
                        dataTime = str(hour * 6)
                        self.frames_data.append(FrameParameters(dataDate, dataTime, is_new))
                        is_new = False

        self.frames = np.asarray(self.frames)
        gc.collect()
        print ('frames shape:')
        print (self.frames.shape)           
        
    def load_frames_default(self):
        self.frames = []
        self.frames_data = []
        
        index = 0
        is_new = True

        for year in self.params.years:
            files = []
            for parameter in self.params.grib_parameters:
                f = open(self.params.grib_folder + parameter + "/" + str(year) + '.grib')
                files.append(f)
                
            while index <= self.params.max_frames:
                gc.collect()
                gids = []
                for f in files:
                    gid = codes_grib_new_from_file(f)
                    gids.append(gid)
                if gids[0] is None:
                    break
                    
                missingVal = codes_get_double(gids[0], 'missingValue')
                    
                # check if this matches our month, year and hour
                dataDate = codes_get(gids[0], 'dataDate')
                dataTime = codes_get(gids[0], 'dataTime')
                include_date, include_time = self.include_datetime(dataDate, dataTime)
                if not (include_date and include_time):
                    is_new = not include_date
                    print 'skipping date: ' + str(dataDate).zfill(8), '            \r',
                    for gid in gids:
                        codes_release(gid)
                    continue
                
                diameter = 1 + 2 * self.params.radius
                    
                frame = np.empty((diameter, diameter, len(self.params.grib_parameters)))  
                gidIt = 0
                for gid in gids:

                    bottomLeft = codes_grib_find_nearest(gid, self.params.start_lat, self.params.start_lon)[0]
                    topRight = codes_grib_find_nearest(gid, self.params.end_lat, self.params.end_lon)[0]
                    
                    latIt = 0
                    for lat in reversed(list(drange(bottomLeft.lat, topRight.lat + GRID_SIZE, GRID_SIZE))):
                        lonIt = 0
                        for lon in drange(bottomLeft.lon, topRight.lon + GRID_SIZE, GRID_SIZE):
                            nearest = codes_grib_find_nearest(gid, lat, lon)[0]
                            frame[latIt, lonIt, gidIt] = max(0, nearest.value)
                            if nearest.value == missingVal:
                                raise Warning('missing value!')

                            lonIt = lonIt + 1
                            
                        latIt = latIt + 1
                            
                    gidIt = gidIt + 1
                
                self.frames.append(frame)
                self.frames_data.append(FrameParameters(dataDate, dataTime, is_new))
                is_new = False 
               
                for gid in gids:
                    codes_release(gid)
                index = index + 1
                print 'loading frames: ', index, '                 \r',

            for f in files:
                f.close()
            if index > self.params.max_frames:
                break
            
        print ('')
        self.frames = np.asarray(self.frames)
        gc.collect()
        print ('frames shape:')
        print (self.frames.shape)
        
    def normalize_frames(self):
        """ 
            Normalizes and reshapes loaded frames 
        """
        self.frames = self.frames.astype('float32')

        nb_params = len(self.params.grib_parameters)
        self.scalers = np.empty((nb_params, 2))
        
        # we normalize per grib  parameter
        for i in range(nb_params):
            array = self.frames[:,:,:,i]
            min_data, ptp_data = array.min(), array.ptp()
            array = (array - min_data) / ptp_data
            self.scalers[i, 0] = min_data
            self.scalers[i, 1] = ptp_data
            self.frames[:,:,:,i] = array
        
    def inverse_transform_data(self, flatten=True):
        """
            @return unscaled (true) dataX and dataY
        """     
        nb_params = len(self.params.grib_parameters)
        dataX = np.empty(self.dataX.shape)

        for i in range(dataX.shape[4]):
            dataX[:,:,:,:,i] = self.dataX[:,:,:,:,i] * self.scalers[i, 1] + self.scalers[i, 0]

        dataY = np.empty(self.dataY.shape)
        for i in range(dataY.shape[1]):
            dataY[:,i] = self.dataY[:,i] * self.scalers[i, 1] + self.scalers[i, 0]
      
        return dataX, dataY
  
    def predict_data(self, model, flatten=True):
        """
            predicts dataY with the given model
            using dataX as input and unscales it
            @param flatten: whether to flatten the reshaped data or leave it seperated per grib parameter
            @return unscaled prediction of shape (nb_samples, features)
        """
        
        predict = model.predict(self.dataX)
        
        nb_params = len(self.params.grib_parameters)
       
        for i in range(predict.shape[1]):
            predict[:,i] = predict[:,i] * self.scalers[i, 1] + self.scalers[i, 0]
                   
        return predict
        
    def predict_constant(self, flatten=True):
        """
            predicts dataY as constant last element from dataX and unscales it
            @param flatten: whether to flatten the reshaped data or leave it seperated per grib parameter
            @return unscaled prediction of shape (nb_samples, features)
        """
        predict = np.empty(self.dataY.shape)
        
        for i in range(predict.shape[2]):
            predict[:,i] = self.dataX[:, -1, self.params.radius, self.params.radius, i]
            predict[:,i] = predict[:,i] * self.scalers[i, 1] + self.scalers[i, 0]

        return predict

    def create_dataset(self, dataset):
        """ 
            convert an array of values into a dataset matrix 
        """
        steps_before = self.params.steps_before
        steps_after = self.params.steps_after
        forecast_distance = self.params.forecast_distance
      
        dataX, dataY = [], []
        for i in range(len(dataset) - steps_before - forecast_distance - steps_after):
            a = dataset[i:(i + steps_before), :, :]
            dataX.append(a)
            dataY.append(dataset[(i + steps_before + forecast_distance), self.params.radius, self.params.radius, :])
        return np.array(dataX), np.array(dataY)
                
class DatasetSquareAreaMultiple(DatasetSquareArea):
    """
        Loads training/test data of a square area around the center from a grid file, shapes and formats it.
        @param params.steps_before: how many frames before does the network take as input (1 default)
        @param params.steps_after: how many frames ahead it should predict. (1 default)
        @param params.forecast_distance: how many frames it should skip when predicting (0 default)
        @param params.max_frames: the maximum frames to load
        @param params.grib_folder: the path to the folder holding grib subfolders
        @param params.grib_parameters: which parameters to consider, for example ['temperature', 'pressure']
        @param params.lon: the longitude of the center point
        @param params.lat: the latitude of the center point
        @param params.radius: the radius around the center
        @param params.years: only load data from those years
        @param params.months: only load data from those months
        @param params.hours: only load data from those hours
    """
    def __init__(self, params):
        Dataset.__init__(self, params)
    
    def inverse_transform_data(self, flatten=True):
        """
            @return unscaled (true) dataX and dataY
        """     
        nb_params = len(self.params.grib_parameters)
        dataX = np.empty(self.dataX.shape)

        for i in range(dataX.shape[4]):
            dataX[:,:,:,:,i] = self.dataX[:,:,:,:,i] * self.scalers[i, 1] + self.scalers[i, 0]

        dataY = np.empty(self.dataY.shape)
        for i in range(dataY.shape[2]):
            dataY[:,:,i] = self.dataY[:,:,i] * self.scalers[i, 1] + self.scalers[i, 0]
      
        return dataX, dataY
  
    def predict_data(self, model, flatten=True):
        """
            predicts dataY with the given model
            using dataX as input and unscales it
            @param flatten: whether to flatten the reshaped data or leave it seperated per grib parameter
            @return unscaled prediction of shape (nb_samples, features)
        """
        
        predict = model.predict(self.dataX)
        
        nb_params = len(self.params.grib_parameters)
       
        for i in range(predict.shape[2]):
            predict[:,:,i] = predict[:,:,i] * self.scalers[i, 1] + self.scalers[i, 0]
                   
        return predict
        
    def predict_constant(self, flatten=True):
        """
            predicts dataY as constant last element from dataX and unscales it
            @param flatten: whether to flatten the reshaped data or leave it seperated per grib parameter
            @return unscaled prediction of shape (nb_samples, features)
        """
        predict = np.empty(self.dataY.shape)
        
        for i in range(predict.shape[2]):
            predict[:,:,i] = self.dataX[:, -1, self.params.radius, self.params.radius, i]
            predict[:,:,i] = predict[:,:,i] * self.scalers[i, 1] + self.scalers[i, 0]

        return predict

    def create_dataset(self, dataset):
        """ 
            convert an array of values into a dataset matrix 
        """
        steps_before = self.params.steps_before
        steps_after = self.params.steps_after
        forecast_distance = self.params.forecast_distance
      
        dataX, dataY = [], []
        for i in range(len(dataset) - steps_before - forecast_distance - steps_after - 1):
            a = dataset[i:(i + steps_before), :, :]
            dataX.append(a)
            dataY.append(dataset[(i + steps_before + forecast_distance):(i + steps_before + forecast_distance + steps_after), self.params.radius, self.params.radius, :])
        return np.array(dataX), np.array(dataY)

