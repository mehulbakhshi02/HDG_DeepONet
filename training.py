import numpy as np
import tensorflow as tf

np.random.seed(10)
tf.random.set_seed(10)

import fileinput
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib

# Build neural network
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.regularizers import l1

# KUNAL 
# I am commenting this line.
# It is creating the weight files in the wrong location

training = False

no_hidden_layers = 3
activation_func = 'tanh'
number_neurons = 50
no_epochs = 300
batch_sz = 4096
val_split = 0.1

def parameter_search(filename):


    # with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
    #     print(file)
    #     for line in file:
    #         print(line)
    f = open(filename, "r")
    for line in f:
        line = line.strip("\n")
        if(line[0:2]=='##'):
            param =  line.split("= ",1)[0]
            param = param[3:-1]

            # print(param)
            if(param=="hidden_layers"):
                no_hidden_layers = int(line.split("= ",1)[1])
            elif(param=="activation_func"):
                activation_func = line.split("= ",1)[1]
            elif(param=='number_neurons'):
                number_neurons = int(line.split("= ",1)[1])
            elif(param=='epochs'):
                no_epochs = int(line.split("= ",1)[1])
            elif(param=='batch_size'):
                batch_sz = int(line.split("= ",1)[1])
            elif(param=='validation_split'):
                val_split = float(line.split("= ",1)[1])
    return [no_hidden_layers, activation_func, number_neurons, no_epochs, batch_sz, val_split]

# KUNAL
# I am commeting the below snippet 

if(training):
    add = ''
else:
  add = "ML/"

# KUNAL
# To prevent the code from breaking down
# add = ''

[no_hidden_layers, activation_func, number_neurons, no_epochs, batch_sz, val_split] = parameter_search(add+"test_ml.pde")

input_data = np.load(add+'All_input_data.npy')
output_data = np.load(add+'All_output_data.npy')

num_data_points = np.shape(input_data)[0]
num_inputs = np.shape(input_data)[1]
num_outputs = np.shape(output_data)[1]

def new_r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred), axis=0)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)
    output_scores =  1 - SS_res / (SS_tot + K.epsilon())
    r2 = K.mean(output_scores)
    return r2

# Define model architecture here
def New_Model(num_inputs):
    field_input = Input(shape=(num_inputs,),name='inputs')
    hidden_layer_1 = Dense(num_inputs,activation='swish')(field_input)
    hidden_layer_1 = Dense(60,activation='swish')(hidden_layer_1)
    hidden_layer_1 = Dense(60,activation='swish')(hidden_layer_1)
    hidden_layer_1 = Dense(70,activation='relu')(hidden_layer_1)
    hidden_layer_1 = Dense(80,activation='relu')(hidden_layer_1)
    hidden_layer_1 = Dense(80,activation='relu')(hidden_layer_1)
    hidden_layer_1 = Dense(100,activation='tanh')(hidden_layer_1)
    hidden_layer_1 = Dense(100,activation='tanh')(hidden_layer_1)
    hidden_layer_1 = Dense(100,activation='tanh')(hidden_layer_1)
    hidden_layer_1 = Dense(90,activation='relu')(hidden_layer_1)
    hidden_layer_1 = Dense(80,activation='relu')(hidden_layer_1)
    hidden_layer_1 = Dense(70,activation='relu')(hidden_layer_1)
    hidden_layer_1 = Dense(50,activation='swish')(hidden_layer_1)
    hidden_layer_1 = Dense(40,activation='swish')(hidden_layer_1)
    hidden_layer_1 = Dense(20,activation='swish')(hidden_layer_1)
    hidden_layer_1 = Dense(10,activation='swish')(hidden_layer_1)
    outputs = Dense(num_outputs,name='outputs')(hidden_layer_1)
    model = Model(inputs=[field_input],outputs=[outputs]) 
    my_adam = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=my_adam,
          loss={'outputs': tf.keras.losses.Huber()},
          metrics=[new_r2])
    return model

# model.summary()
# Optimization
weights_filepath = 'best_weights.weights.h5'
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
callbacks_list = [checkpoint,earlystopping]

if __name__ == '__main__':

    idx = np.arange(num_data_points)
    np.random.shuffle(idx)

    input_data = input_data[idx]
    output_data = output_data[idx]
    # Preprocessing
    preproc_input = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
    input_data = preproc_input.fit_transform(input_data)
    scaler_filename = add+f"ip_scaler.save"
    joblib.dump(preproc_input, scaler_filename)

    preproc_output = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])

    # KUNAL
    # removing the inf values:
    inf_index = np.argwhere(np.isinf(output_data))
    output_data = np.delete(output_data,inf_index,axis=0)
    input_data = np.delete(input_data,inf_index,axis=0)
    # ends here

    output_data = preproc_output.fit_transform(output_data)
    scaler_filename = add+f"op_scaler.save"
    joblib.dump(preproc_output, scaler_filename)
    
    num_stack_model = 3
    # prediction = 0
    # no_epochs = 1
    for i in range(num_stack_model):
        if i > 0:
            num_inputs = np.shape(output_data)[1]
            input_data = prediction

        model = New_Model(num_inputs)
    
        weights_filepath = f'best_weights_{i}.weights.h5'
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        callbacks_list = [checkpoint,earlystopping]

        model.fit(x=input_data,y=output_data,
          epochs=no_epochs,
          batch_size=batch_sz,validation_split=val_split,callbacks=callbacks_list)

        model.load_weights(weights_filepath)
        prediction = model.predict(input_data)
        
        del model
