import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
     
        def sigmoid(x):
            return 1 / (1 + np.exp(-x)) # Replace 0 with your sigmoid calculation here
        self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o)
            
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        ### Forward pass ###
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        """
        The hidden layer will use the sigmoid function for activations. 
        The output layer has only one node and is used for the regression, 
        the output of the node is the same as the input of the node. 
        That is, the activation function is  f(x)= x
        """
        final_outputs = final_inputs #self.activation_function(final_inputs) # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
 
        ### Backward pass ###

        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        # hidden_error = None

        # output term = (y-y_hat)f'(weights_hidden_output.hidden_layer_output) = error * f'(output_layer_in)
        # Now f'(output_layer_in) = f(output_layer_in) * (1 - f(output_layer_in)) = final_outputs * (1 - final_outputs)
        #output_error_term = error * (final_outputs * (1 - final_outputs))

        output_error_term = error # This is because f(x) = x for output layer
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        

        beta = (1 - hidden_outputs)
        beta_prime = beta * hidden_outputs
        hidden_error_term = hidden_error * beta_prime
        # Weight step (hidden to output)
        # *****************************************************************************************************
        # delta_weights_h_o = delta_weights_h_o + hidden_outputs * output_error_term  ->throws below
        # numpy error 
        # ValueError: non-broadcastable output operand with shape (2,1) 
        # doesn't match the broadcast shape (2,2)
        # ******************************************************************************************************
        delta_weights_h_o = delta_weights_h_o + output_error_term * hidden_outputs[:,None]
        
        # Weight step (input to hidden)
        delta_weights_i_h = delta_weights_i_h + hidden_error_term * X[:,None]
        
       
        
        #print("hidden_outputs - {}".format(hidden_outputs))
        #print("output_error_term - {}".format(output_error_term))
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        
        self.weights_hidden_to_output = self.weights_hidden_to_output + self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden = self.weights_input_to_hidden + self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        #final_outputs = self.activation_function(final_inputs) 
        final_outputs = final_inputs # signals from final output layer # This is because f(x) = x for out put layer

        
        return final_outputs


#########################################################
# hyperparameters 
##########################################################
iterations = 4000
learning_rate = 0.5
hidden_nodes = 22
output_nodes = 1
