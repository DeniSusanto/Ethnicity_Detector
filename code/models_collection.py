import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import BatchNormalization, Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, AveragePooling2D, ZeroPadding2D, Concatenate, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping #prevent losing model training since my computer shuts down alot when overheat
from tensorflow.keras.utils import to_categorical, Sequence

MODEL_INPUT_SHAPE = (256, 256, 3)

def create_vgg_custom_net_1():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v2():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(32, activation = "softmax"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v3():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(100, activation = "relu"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v4():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(1000, activation = "relu"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v5():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(256, activation = "relu"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v6():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    
    vgg_custom_net.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(64, activation = "relu"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v7():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(512, activation = "relu"))
    vgg_custom_net.add(Dense(256, activation = "relu"))
    vgg_custom_net.add(Dense(128, activation = "relu"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v8():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(256, activation = "relu"))
    vgg_custom_net.add(Dense(128, activation = "relu"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v9():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(64, activation = "relu"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v10():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(32, activation = "relu"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v11():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(32, activation = "relu"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v12():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    
    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(16, activation = "relu"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v13():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(16, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(16, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    
    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v14():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    
    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v15():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(8, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(16, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    
    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v16():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(8, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    
    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v17():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))

    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    
    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v18():
    vgg_custom_net = Sequential()
 
    vgg_custom_net.add(Conv2D(16, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
  
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    
    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_1_v19():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(8, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    
    vgg_custom_net.add(Conv2D(8, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    
    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(16, activation = "softmax"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net
#2nd
def create_vgg_custom_net_2():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(512, activation = "softmax"))
    vgg_custom_net.add(Dense(64, activation = "softmax"))
    vgg_custom_net.add(Dense(10, activation = "softmax"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_3():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(5, activation = "softmax"))

    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_vgg_custom_net_3_v2():
    vgg_custom_net = Sequential()
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    
    vgg_custom_net.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))
    vgg_custom_net.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    vgg_custom_net.add(BatchNormalization())
    vgg_custom_net.add(Dropout(0.4))
    vgg_custom_net.add(MaxPooling2D((2, 2)))

    vgg_custom_net.add(Flatten())
    vgg_custom_net.add(Dense(128, activation = "softmax"))
    vgg_custom_net.add(Dense(64, activation = "softmax"))
    vgg_custom_net.add(Dense(5, activation = "softmax"))
    
    vgg_custom_net.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return vgg_custom_net

def create_AlexNet_original():
    model = Sequential()
    
    model.add(Conv2D(96, (11, 11), strides = 4, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3), strides = 2))
    
    model.add(ZeroPadding2D(padding=(2, 2)))    
    model.add(Conv2D(256, (5, 5), strides = 1, padding = 'valid', activation = "relu"))
    model.add(MaxPooling2D((3, 3), strides = 2))
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(384, (3, 3), strides = 1, padding = 'valid', activation = "relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(384, (3, 3), strides = 1, padding = 'valid', activation = "relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), strides = 1, padding = 'valid', activation = "relu"))
    model.add(MaxPooling2D((3, 3), strides = 2))
    
    model.add(Flatten())
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(1000, activation = "softmax"))
    model.add(Dense(5, activation = "softmax"))
    
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_original_v2():
    model = Sequential()
    
    model.add(Conv2D(96, (11, 11), strides = 4, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3), strides = 2))
    
    model.add(ZeroPadding2D(padding=(2, 2)))    
    model.add(Conv2D(256, (5, 5), strides = 1, padding = 'valid', activation = "relu"))
    model.add(MaxPooling2D((3, 3), strides = 2))
    
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(384, (3, 3), strides = 1, padding = 'valid', activation = "relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(384, (3, 3), strides = 1, padding = 'valid', activation = "relu"))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(256, (3, 3), strides = 1, padding = 'valid', activation = "relu"))
    model.add(MaxPooling2D((3, 3), strides = 2))
    
    model.add(Flatten())
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(1000, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))
    
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized():
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Flatten())
    model.add(Dense(4096, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(1000, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v2():
    #Optimized AlexNet with Batch Normalization
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Flatten())
    model.add(Dense(4096, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(1000, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v3():
    #Optimized AlexNet with Batch Normalization
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(50, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v4():
    #Optimized AlexNet with Batch Normalization
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v5():
    #Optimized AlexNet with Batch Normalization
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v6():
    #Optimized AlexNet with Batch Normalization
    model = Sequential()
    
    model.add(Conv2D(32, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(64, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v7():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v8():
    #Optimized AlexNet with Batch Normalization
    model = Sequential()
    
    model.add(Conv2D(32, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(64, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v9():
    #Optimized AlexNet with Batch Normalization
    model = Sequential()
    
    model.add(Conv2D(32, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(64, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v10():
    #Optimized AlexNet with Batch Normalization
    model = Sequential()
    
    model.add(Conv2D(32, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(64, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v11():
    #Optimized AlexNet with Batch Normalization
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(16, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v12():
    #Optimized AlexNet with Batch Normalization
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v13():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v14():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v15():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(64, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v16():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(16, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v17():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(32, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(64, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(16, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v18():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v19():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v20():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(64, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(32, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v21():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (19, 19), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (11, 11), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(64, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(32, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v22():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v23():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v24():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v25():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(16, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(32, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(16, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v26():
    #using v5 architecture with changed LR
    model = Sequential()
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v27():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v28():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v29():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(8, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(16, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(16, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v30():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(8, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(8, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(16, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v31():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(8, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(8, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(16, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v32():
    #using v5 architecture with changed LR
    model = Sequential()
    
    model.add(Conv2D(4, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(8, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(8, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v33():
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v34():
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "softmax"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v35():
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Flatten())
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(1000, activation = "softmax"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v36():
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Flatten())
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(1000, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v37():
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "softmax"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v38():
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(192, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(1000, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v39():
    model = Sequential()
    
    model.add(Conv2D(32, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(64, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(64, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(4096, activation = "relu"))
    model.add(Dense(1000, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v40():
    model = Sequential()
    
    model.add(Conv2D(8, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(16, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(32, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dense(256, activation = "relu"))
    model.add(Dense(64, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_custom_net_1():
    #Model with decent number of parameters, progressively smaller filter, batch normalization, and dropout
    model = Sequential()
    model.add(Conv2D(32, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(32, (11, 11), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (5, 5), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (5, 5), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(512, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(1000, activation = "softmax"))
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_custom_net_2():
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(BatchNormalization())
    model.add(Dense(5, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_LeNet5():
    model = Sequential()

    model.add(Conv2D(6, (3, 3), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(AveragePooling2D())
    
    model.add(Conv2D(16, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(AveragePooling2D())
    
    model.add(Flatten())
    model.add(Dense(120, activation = "relu"))
    model.add(Dense(84, activation = "relu"))
    model.add(Dense(5, activation = "softmax"))
    
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def create_AlexNet_optimized_v34_no_others():
    model = Sequential()
    
    model.add(Conv2D(64, (11, 11), strides = 1, padding = "same", activation = "relu", input_shape = MODEL_INPUT_SHAPE))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Conv2D(128, (7, 7), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(128, (3, 3), strides = 1, padding = "same", activation = "relu"))
    model.add(MaxPooling2D((3, 3)))
    
    model.add(Flatten())
    model.add(Dense(16, activation = "softmax"))
    model.add(Dense(4, activation = "softmax"))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model