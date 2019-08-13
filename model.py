from keras.layers import Input, Dense, Lambda, Conv2D, Dropout, Flatten
from keras.models import Model 
from keras.optimizers import Adam
import numpy as np
import cv2
#def load_data()

def pre_process(image):
    #first the sky and the car could be remove from the image to make it smaller
    image = image[60:130, :, :]
    image = cv2.resize(image, (200,70), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def build_model():
    inputs = Input(shape=(70,200,3))

    norm = Lambda(lambda x: x/127.5-1.0)(inputs)

    convs = Conv2D(24, kernel_size=(5,5), strides=(2,2), activation="relu")(norm)
    convs = Conv2D(36, kernel_size=(5,5), strides=(2,2), activation="relu")(convs)
    convs = Conv2D(48, kernel_size=(5,5), strides=(2,2), activation="relu")(convs)
    convs = Conv2D(64, kernel_size=(3,3), activation="relu")(convs)
    convs = Conv2D(64, kernel_size=(3,3), activation="relu")(convs)
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


#---------- Main ----------
if __name__ == '__main__':
    #data_input = load_data()

    model = build_model()
    model.save('no_train.h5')
    #train_model()
    #im = np.zeros((66,200,3))
    #a = im[60:140,:]
    #im = pre_process(im)
    #print(a.shape)