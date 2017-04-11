"""
   Loading UV
"""
import os, math, re, gc, sys
import decimal
import numpy as np
import scipy as sp

from attrdict import AttrDict

from eccodes import *
import colormap as cm
import matplotlib.pyplot as plt

GRID_SIZE = 0.75
#GRIB_FOLDER = '/home/isa/sftp/'
GRIB_FOLDER = '/media/isa/VIS1/'

def load_data(subfolder, year):
    """ 
        Loads data from the grib file 
        and returns it
    """
    f = open(GRIB_FOLDER + subfolder + "/" + str(year) + '.grib')
    frame_it = 0
    images = []
    while 1:
        frame_it = frame_it + 1
        gid = codes_grib_new_from_file(f)
        if gid is None:
            break

        missingVal = codes_get_double(gid, 'missingValue')

        img = np.empty([241,480], dtype=float)
        iterid = codes_grib_iterator_new(gid, 0)

        while 1:
            result = codes_grib_iterator_next(iterid)
            if not result:
                break
 
            [lat, lon, value] = result
            img[int((-lat + 90) / GRID_SIZE), int(lon / GRID_SIZE)] = value
         
            if value == missingVal:
                print "missing"

        images.append(img)
        codes_grib_iterator_delete(iterid)
        codes_release(gid)

    f.close()
    images = np.asarray(images)
    images = images.astype('float32')
    return images
    
def plot_uv(images):
    colmap = cm.RdYlGn_r(10)
    
    image_it = 0
    for image in images:
        plt.imshow(image, cmap=colmap, interpolation='none', extent=[0, 360, -90, 90])
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Wind V Component (' + str(image_it) + ')')
        plt.savefig('uv/image_v_' + str(image_it) + '.png')
        cb = plt.colorbar()
        cb.set_label('Wind speed (v component)', rotation=270, labelpad=20)
        plt.close('all')
        image_it = image_it + 1
        
def bla(year):
    images_u = load_data('wind_u', year)
    images_v = load_data('wind_v', year)
   
    print(images_u.shape)
    print(images_v.shape) # time x 241 (lat) x 480 (lon)
    
    resZ = images_u.shape[0]
    resY = images_u.shape[2]
    resX = images_u.shape[1]
    
    data = np.empty((resZ * resY * resX * 2))

    for z in range(resZ):
        for y in range(resY):
            for x in range(resX):
                index = z * resY * resX + y*resX + x
                data[2 * index + 0] = images_v[z, x, y]
                data[2 * index + 1] = images_u[z, x, y]

    print(data.shape)
    data = data.astype('float32')
    data.tofile(GRIB_FOLDER + 'uv/' + str(year) + '.bin')

def main():
    print(sys.byteorder)

    for year in range(1986, 2017):
        bla(year)
    return 1

if __name__ == "__main__":
    sys.exit(main())

