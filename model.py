from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv3D, Input, MaxPool3D, Flatten, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model


def c3d_model(frames, width, height, channels):
    input_shape = (frames, height, width, channels)
    weight_decay = 0.005
    # nb_classes = 2

    inputs = Input(input_shape)
    x = Conv3D(8, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    # x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same', activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    #
    # x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same', activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    #
    # x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same', activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)
    #
    # x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(16, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    # x = Dense(2048,activation='relu', kernel_regularizer=l2(weight_decay))(x)
    # x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs, x)

    return model
