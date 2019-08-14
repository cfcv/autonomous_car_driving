from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Lambda, Conv2D, Dropout, Flatten, BatchNormalization, Activation
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.activations import relu

import numpy as np
import cv2
import pandas as pd
import training_utils as TU


def load_data(path_to_csv):
    #Read the csv generated by the udacity simulator on training mode
    data = pd.read_csv(path_to_csv, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    #Get the input data to the CNN
    images_input = data[['center', 'left', 'right']].values

    #Get the expected output(Steering angle) of the CNN
    output_steering = data['steering'].values

    #Split the data into train and validation. train ~= 80% and validation ~= 20%
    input_training, input_valid, output_training, output_valid = train_test_split(images_input, output_steering, test_size=686)

    return input_training, input_valid, output_training, output_valid

def build_model():

    #PilotNET
    inputs = Input(shape=(70,200,3))

    norm = Lambda(lambda x: x/127.5-1.0)(inputs)

    convs = Conv2D(24, kernel_size=(5,5), strides=(2,2))(norm)
    convs = BatchNormalization()(convs)
    convs = Activation('relu')(convs)

    convs = Conv2D(36, kernel_size=(5,5), strides=(2,2))(convs)
    convs = BatchNormalization()(convs)
    convs = Activation('relu')(convs)

    convs = Conv2D(48, kernel_size=(5,5), strides=(2,2))(convs)
    convs = BatchNormalization()(convs)
    convs = Activation('relu')(convs)

    convs = Conv2D(64, kernel_size=(3,3))(convs)
    convs = BatchNormalization()(convs)
    convs = Activation('relu')(convs)

    convs = Conv2D(64, kernel_size=(3,3))(convs)
    convs = BatchNormalization()(convs)
    convs = Activation('relu')(convs)

    drop = Dropout(0.5)(convs)
    f = Flatten()(drop)
    x = Dense(100, activation='relu')(f)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    output = Dense(1)(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='mean_squared_error', optimizer=Adam())
    model.summary()

    return model

def train_model(model, X_train, X_valid, Y_train, Y_valid):
    #Save only the best model in the validation loss
    checkpoint = ModelCheckpoint('track2_30_epochs.h5', monitor='val_loss', save_best_only=True)

    #Some hyperparameters
    epochs = 30
    batch_size = 128
    it_per_epoch = np.ceil(len(X_train) / batch_size)

    model.fit_generator(TU.batch_generator(X_train, Y_train, batch_size, is_training=True),
                        steps_per_epoch=it_per_epoch,
                        epochs=epochs,
                        max_q_size=1,
                        callbacks=[checkpoint],
                        validation_data=TU.batch_generator(X_valid, Y_valid, batch_size, is_training=False),
                        nb_val_samples=len(X_valid),
                        verbose=2)


#---------- Main ----------
if __name__ == '__main__':
    X_train, X_valid, Y_train, Y_valid = load_data('driving_log.csv')

    model = build_model()
    #model.save('no_train.h5')
    #model = load_model('no_train.h5')

    train_model(model, X_train, X_valid, Y_train, Y_valid)
    #im = np.zeros((66,200,3))
    #a = im[60:140,:]
    #im = pre_process(im)
    #print(a.shape)