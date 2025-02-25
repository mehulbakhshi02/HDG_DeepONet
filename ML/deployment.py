import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def process_data(trans_adj_coordinates, norm_u_sol):

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

    # Hard coded X_test generation

    num_output_points = 45
    num_sensors = 28
    num_samples = int(len(norm_u_sol)/28)  #number of elements

    branch_input = np.zeros((num_samples, num_output_points, num_sensors))
    trunk_input = np.zeros((num_samples, num_output_points, 2))
    ml_adjoint = np.zeros((num_samples, num_output_points))

    rr = 0

    for i in range(num_samples):
        for k in range(num_sensors):
            branch_input[i, 0, k] = norm_u_sol[rr]
            rr = rr+1
        branch_input[i, :, :] = branch_input[i, 0, :]
    
    kk = 0
    for j in range(num_output_points):
        for k in range(2):
            trunk_input[0, j, k] = trans_adj_coordinates[kk]
            kk = kk+1
    trunk_input[:, :, :] = trunk_input[0, :, :]

    X_test = (branch_input, trunk_input)

    def new_r2(y_true, y_pred):
        SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
        SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        output_scores =  1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
        r2 = tf.reduce_mean(output_scores)
        return r2

    model_deepon_loaded = tf.keras.models.load_model("deeponet_model.keras", custom_objects={"new_r2": new_r2, "Deepon": Deepon, "FNN": FNN})

    ml_adjoint = model_deepon_loaded.predict(X_test)

    ml_adjoint = ml_adjoint.tolist()

    return ml_adjoint