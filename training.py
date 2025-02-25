import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

tf.keras.utils.get_custom_objects().clear()

@tf.keras.utils.register_keras_serializable()
class FNN(tf.keras.Model):
    # Initialization
    def __init__(self, layer, activation, **kwargs):
        super(FNN, self).__init__(**kwargs)
        self.layer = layer
        # This loop creates a list of hidden layers. The loop iterates over layer (except the last element, which is the output layer)
        num_h = len(layer)
        self.branch_hidden = []
        for i in range(num_h-1):
            self.branch_hidden.append(tf.keras.layers.Dense(units=layer[i], activation=activation, kernel_initializer='glorot_uniform', bias_initializer='random_normal'))
        # The last layer (layer[-1]) is the output layer. It does not use an activation function, as this is typical for regression tasks or when applying the activation function later
        self.branch_out = tf.keras.layers.Dense(units=layer[-1], kernel_initializer='glorot_uniform', bias_initializer='random_normal')
    
    # Forward Passing
    def call(self, input):
        x = input
        L = len(self.layer)
        # A loop runs over all the hidden layers (L-1 times) and applies them sequentially. Each layer processes the data and passes the output to the next layer.
        for i in range(L-1):
            x = self.branch_hidden[i](x)
        return self.branch_out(x) # returning model's prediction

    def get_config(self):
        base_config = super().get_config()
        config = {
            "layer": self.layer,
            "activation": self.branch_hidden[0].activation,  # Assuming all hidden layers have the same activation
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        layer = config.pop("layer")
        activation = config.pop("activation")
        return cls(layer=layer, activation=activation, **config)

@tf.keras.utils.register_keras_serializable()
class Deepon(tf.keras.Model):
    def __init__(self, branch_layer, trunk_layer, activation, **kwargs):
        super(Deepon, self).__init__(**kwargs)
        self.branch = FNN(layer=branch_layer, activation=activation)
        self.trunk = FNN(layer=trunk_layer, activation=activation)
        self.bias = tf.Variable(initial_value=tf.zeros((1,)), trainable=True, name='output_bias')

    def call(self, input):
        input_branch = input[0]
        input_trunk = input[1]
        output_branch = self.branch(input_branch)
        ouput_trunk = self.trunk(input_trunk)
        return tf.reduce_sum(tf.multiply(output_branch, ouput_trunk),-1) + self.bias

    def get_config(self):
        base_config = super().get_config()
        config = {
            "branch_layer": self.branch.layer,
            "trunk_layer": self.trunk.layer,
            "activation": self.branch.branch_hidden[0].activation,  # Assuming all hidden layers have the same activation
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        branch_layer = config.pop("branch_layer")
        trunk_layer = config.pop("trunk_layer")
        activation = config.pop("activation")
        return cls(branch_layer=branch_layer, trunk_layer=trunk_layer, activation=activation, **config)
    
def generate_data():
    branch_input_array = np.load('ML/branch_input.npy')
    trunk_input_array = np.load('ML/trunk_input.npy')
    output_array = np.load('ML/output.npy')
    
    X_train = (branch_input_array, trunk_input_array)
    y_train = output_array

    return (X_train, y_train)

X_train, y_train = generate_data()
branch_layer = [30, 40, 50, 60]
trunk_layer  = [10, 20, 30, 60]
activation = 'tanh'
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
num_epochs = 300
batch_size = 10000

print(X_train[0].shape)
print(X_train[1].shape)
print(y_train.shape)

def new_r2(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    output_scores =  1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    r2 = tf.reduce_mean(output_scores)
    return r2

model_deepon = Deepon(branch_layer=branch_layer, trunk_layer=trunk_layer, activation=activation)
model_deepon.compile(optimizer, loss=tf.keras.losses.Huber(), metrics=[new_r2])
model_deepon.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.01, verbose=1)

model_deepon.save("deeponet_model.keras")
