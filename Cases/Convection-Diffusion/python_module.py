import numpy as np
np.random.seed(10)
import tensorflow as tf
tf.random.set_seed(10)
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib

import fileinput
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Build neural network
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.regularizers import l1

# KUNAL
import subprocess

print("Copying the training.py from the parent directory: Initiated")
subprocess.call('cp training.py ./ML/training.py', shell=True)
print("Copying the training.py from the parent directory: Completed")  


# from training import model, weights_filepath

# model.load_weights("ML/"+weights_filepath)
scaler_filename = "ML/ip_scaler.save"
preproc_input = joblib.load(scaler_filename)

scaler_filename = "ML/op_scaler.save"
preproc_output = joblib.load(scaler_filename)
# Define model architecture here

def new_r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred), axis=0)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)
    output_scores =  1 - SS_res / (SS_tot + K.epsilon())
    r2 = K.mean(output_scores)
    return r2
    
def New_Model(num_inputs):
    num_outputs = 1
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
          loss_weights=[1.0], metrics=[new_r2])
    return model

def collection_func():
    print("Hello world")

def output_ml_error_prediction(input_value):
    # Modifying to check if we should predict or not depending on the input given to ML_Option.txt
    
    file = open("ML_Option.txt")
    ML_Option = file.readlines()
    # print(f"python_module.py")
    if int(ML_Option[0][0]) == 1:
        # print(f"ML Deployed: ")
        input_value = input_value
        print(input_value.shape)
        input_value = preproc_input.transform(input_value)
   
        # output_value = model.predict(input_value)
        # print("Output in python")
    
        num_stack_model = 3
        # prediction = 0
        # no_epochs = 1
        for i in range(num_stack_model):
            num_inputs = 60
            if i > 0:
                num_inputs = 1
            model = New_Model(num_inputs)
            model.load_weights("ML/"+f'best_weights_{i}.weights.h5')
            input_value = model.predict(input_value)
            # print(preproc_output.inverse_transform(input_value))
        output_value = input_value
        output_value = preproc_output.inverse_transform(output_value)
        # print(output_value.shape)
    else:
        # print(f"ML NOT Deployed: ")
        # Hard coded
        output_value = np.zeros((1,1))
    return output_value.astype('float64') # Tensorflow causes cast to float32 - this line reverses it
   
if __name__ == '__main__':
    # print(output_ml_error_prediction(np.zeros(shape=(10,54))))
    pass

print("Removing the training.py from the ./ML: Initiated")
subprocess.call('rm ./ML/training.py', shell=True)
print("Removing the training.py from the ./ML: Completed")  

