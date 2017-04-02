from attrdict import AttrDict
from keras.models import load_model
import numpy as np
np.set_seed
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed

EPOCHS = 10

def create_dataset():
    """
        Creates a dataset with 
        dataX and dataY for training/testing/validation
        with parameters according to the baseline paper.
    """    
    
    

def create_model(steps_before, steps_after, feature_count):
    """ 
        creates, compiles and returns a RNN model 
        @param steps_before: the number of previous time steps (input)
        @param steps_after: the number of posterior time steps (output or predictions)
        @param feature_count: the number of features in the model
        @param hidden_neurons: the number of hidden neurons per LSTM layer
    """
    DROPOUT = 0.5
    LAYERS = 2
    
    hidden_neurons = 300
    '''
    model = Sequential()  
    model.add(LSTM(input_dim=feature_count, output_dim=hidden_neurons, return_sequences=False))  
    model.add(Dense(feature_count))
    model.add(Activation('linear')) '''  
    
    model = Sequential()
    model.add(LSTM(input_dim=feature_count, output_dim=hidden_neurons, return_sequences=False))
    model.add(RepeatVector(steps_after))
    model.add(LSTM(output_dim=hidden_neurons, return_sequences=True))
    #model.add(LSTM(input_dim=feature_count, output_dim=hidden_neurons, return_sequences=True))
    model.add(TimeDistributed(Dense(feature_count)))
    model.add(Activation('linear'))  
    
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])  
    return model

def train_model(model, dataX, dataY, epoch_count, model_folder):
    """ 
        trains only the sinus model
    """
    history = model.fit(dataX, dataY, batch_size=1, nb_epoch=epoch_count, validation_split=0.05)
    
    '''if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # save model
    model.save(model_folder + '/model.h5')
  
    # plot training and val loss/accuracy
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(model_folder + '/history_acc.png')
    plt.cla()    

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(model_folder + '/history_loss.png')
    plt.cla()'''

def test_model():
    ''' 
        testing how well the network can predict
        a simple sinus wave.
    '''
    t = np.arange(0.0, 4.0, 0.02)
    sinus = np.sin(2 * np.pi * t)
    sinus = sinus.reshape((sinus.shape[0], 1))
    n_pre = 50
    n_post = 10
    
    dX, dY = [], []
    for i in range(len(sinus)-n_pre-n_post):
        dX.append(sinus[i:i+n_pre])
        dY.append(sinus[i+n_pre:i+n_pre+n_post])
        #dY.append(sinus[i+n_pre])
    dataX = np.array(dX)
    dataY = np.array(dY)

    # create and fit the LSTM network
    print('creating model...')
    model = create_model(n_pre, n_post, 1)
    train_sinus(model, dataX, dataY, EPOCHS, 'test_sinus/model')
    
    # now test
    t = np.arange(15.0, 19.0, 0.02)
    sinus = np.sin(2 * np.pi * t)
    sinus = sinus.reshape((sinus.shape[0], 1))
    
    dX, dY = [], []
    for i in range(len(sinus)-n_pre-n_post):
        dX.append(sinus[i:i+n_pre])
        dY.append(sinus[i+n_pre:i+n_pre+n_post])
        #dY.append(sinus[i+n_pre])
    dataX = np.array(dX)
    dataY = np.array(dY)
    
    predict = model.predict(dataX)
    
    # now plot
    nan_array = np.empty((n_pre - 1))
    nan_array.fill(np.nan)
    nan_array2 = np.empty(n_post)
    nan_array2.fill(np.nan)
    ind = np.arange(n_pre + n_post)

    fig, ax = plt.subplots()
    for i in range(0, 50, 50):

        forecasts = np.concatenate((nan_array, dataX[i, -1:, 0], predict[i, :, 0]))
        ground_truth = np.concatenate((nan_array, dataX[i, -1:, 0], dataY[i, :, 0]))
        #forecasts = np.concatenate((nan_array, dataX[i, -1:, 0], predict[i, :]))
        #ground_truth = np.concatenate((nan_array, dataX[i, -1:, 0], dataY[i, :]))
        network_input = np.concatenate((dataX[i, :, 0], nan_array2))
     
        ax.plot(ind, network_input, 'b-x', label='Network input')
        ax.plot(ind, forecasts, 'r-x', label='Many to many model forecast')
        ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')
        
        plt.xlabel('t')
        plt.ylabel('sin(t)')
        plt.title('Sinus Many to Many Forecast')
        plt.legend(loc='best')
        plt.savefig('test_sinus/plot_mtm_triple_' + str(i) + '.png')
        plt.cla()

def main():
    test_sinus()
    return 1

if __name__ == "__main__":
    sys.exit(main())
