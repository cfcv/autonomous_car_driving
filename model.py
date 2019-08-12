from keras.layers import Input, Dense, Lambda, Conv2D, Dropout
from keras.models import Model 
#def load_data()

def build_model():
    inputs = Input(shape=(66,200,3))

    norm = Lambda(lambda x: x/127.5-1.0)(inputs)

    convs = Conv2D(24, kernel_size=(5,5), strides=(2,2))(norm)
    convs = Conv2D(36, kernel_size=(5,5), strides=(2,2))(convs)
    convs = Conv2D(48, kernel_size=(5,5), strides=(2,2))(convs)
    convs = Conv2D(64, kernel_size=(3,3))(convs)
    convs = Conv2D(64, kernel_size=(3,3))(convs)
    drop = Dropout(0.5)(convs)
    f = Flatten()(drop)
    x = Dense(100, activation='relu')(f)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    output = Dense(1)(x)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model


#---------- Main ----------
if __name__ == '__main__':
    #data_input = load_data()

    model = build_model()

    #train_model()