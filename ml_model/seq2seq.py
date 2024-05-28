import tensorflow as tf 
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Reshape, Concatenate



def get_seq2seq_model(num_obs, num_orders, horizons, latent_dim):

    # input_train = keras.layers.Input(shape=(num_obs, 4*num_orders))

    
    ############### Seq2Seq ###############

    encoder_inputs = Input(shape=(num_obs, 4*num_orders))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_output, state_h, state_c = encoder(encoder_inputs)         # encoder_output and state_h are the same
    states = [state_h, state_c]

    decoder_input = keras.layers.Input(shape=(1, 3))
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_dense = keras.layers.Dense(3, activation='softmax')

    ########################################

    all_outputs = []
    encoder_output = keras.layers.Reshape((1, int(encoder_output.shape[1])))(encoder_output)
    inputs = keras.layers.Concatenate(axis=2)([decoder_input, encoder_output])

    
    for _ in range(horizons):
        
        output, state_h, state_c = decoder_lstm(inputs, initial_state=states)


        #### Make Prediction ####

        prediction = decoder_dense(Concatenate(axis=2)([output, encoder_output]))
        all_outputs.append(prediction)
        
        #########################
        
        inputs = keras.layers.Concatenate(axis=2)([prediction, encoder_output])
        states = [state_h, state_c]

        
    
    combined_outputs = keras.layers.Concatenate(axis=1)(all_outputs)
    
    model = keras.models.Model([encoder_inputs, decoder_input], combined_outputs)
    return model