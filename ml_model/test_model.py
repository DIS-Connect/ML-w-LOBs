import tensorflow as tf 
from tensorflow import keras
from keras import backend as K


tf.config.run_functions_eagerly(True)

def get_model(num_obs, num_orders, horizons):

    input_train = keras.layers.Input(shape=(num_obs, 4*num_orders))
    
    
    flat = keras.layers.Flatten()(input_train)
    
    
    
    #dense1 = keras.layers.Dense(4096, activation='sigmoid')(flat)
    #dense2 = keras.layers.Dense(8192, activation='relu')(dense1)
    dense3 = keras.layers.Dense(4096, activation='relu')(flat)
    
    
    outputs = []
    
    for i in range(horizons):
        dense4 = keras.layers.Dense(4096, activation='relu')(dense3)
        prediction = keras.layers.Dense(3, activation='softmax')(dense4)
        prediction = tf.expand_dims(prediction, axis=1)  # Add a new axis
        outputs.append(prediction)
        
        
    combined_outputs = keras.layers.Concatenate(axis=1)(outputs) 
    
    model = keras.models.Model(input_train, combined_outputs)
    return model