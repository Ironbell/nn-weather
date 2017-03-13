from attrdict import AttrDict

from dataset import *
from network import *

GRIB_TEMPERATURE = '/home/isa/sftp/total_2mtemp.grib'
GRIB_PRESSURE = '/home/isa/sftp/total_spressure.grib'

def run():
    window_size = 5
    epoch_count = 10
    params = AttrDict()
    params.window_size = window_size
    params.start_lat = 45.839803
    params.end_lat = 47.749943
    params.start_lon = 6.108398 + 180
    params.end_lon = 10.524902 + 180
    params.grib_file = GRIB_TEMPERATURE
    params.max_frames = 1000
    params.test_fraction = 0.3
    params.years = [2000, 2001, 2002, 2003]
    params.months = [6, 7, 8]

    # load the data from the .grib files
    data = DatasetMultiple(params)
    
    # create and fit the LSTM network
    if (True):
        model = create_model(window_size, data.vector_size)
        train_model(model, data, epoch_count)
        model.save('model.h5')
    else:
        model = load_model('model.h5')
   
    # make predictions
    trainPredict, testPredict = evaluate_model(model, data)
    
    # plot the results
    plot_predictions_images(data, trainPredict, testPredict)

def main():
    run()
    return 1

if __name__ == "__main__":
    sys.exit(main())
