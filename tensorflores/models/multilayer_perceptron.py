import numpy as np
import pandas as pd
from river import stream
import json  

from tensorflores.utils.array_manipulation import ArrayManipulation
from tensorflores.utils.cpp_generation import CppGeneration
from tensorflores.utils.json_handle import JsonHandle

class MultilayerPerceptron:
    def __init__(self, 
                 input_size:int = 5,
                 hidden_layer_sizes:list = [64, 32],
                 output_size:int = 1, 
                 activation_functions:list = ['relu', 'relu', 'linear'],
                 weight_bias_init:str = 'RandomNormal',
                 training_with_quantization: bool = False):
        """
        Initialises a neural network with the option of choosing different initialisation methods.

        Args:
            input_size (int): Size of the input layer.
            hidden_layer_sizes (list): List with the size of the hidden layers.
            output_size (int): Size of the output layer.
            activation_functions (list): Activation functions for each layer.
            weight_bias_init (str): Weight initialisation method (‘RandomNormal’, ‘RandomUniform’,‘GlorotUniform’, ‘HeNormal’).
            training_with_quantization (bool): Whether to use quantization during training (default: False).

        Example:

        MLP with 3 input, 2 hidden layers with "ReLu" activations and 1 output layer with "linear" activation

        tf = MultilayerPerceptron(input_size = 3,
                                hidden_layer_sizes = [18, 9],
                                output_size = 1,
                                activation_functions = ['relu', 'relu', 'linear'],
                                weight_bias_init = 'HeNormal')

        
        
        """
       
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.activation_functions = activation_functions
        self.training_with_quantization = training_with_quantization
        self.loss_cumulative = []
        self.validation_loss_cumulative = []
        self.bias_cluster_cumulative = []
        self.weight_cluster_cumulative = []
        self.weight_center = []
        self.bias_center = []
        self.weight_index = []
        self.bias_index = []


        # Lista de tamanhos das camadas
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        # Inicializa pesos e biases com o método escolhido
        self.weights, self.biases = self.__initialize_weights(layer_sizes, weight_bias_init), self.__initialize_biases(layer_sizes)

        # Quantidades de pesos e biases
        self.weights_len = len(ArrayManipulation().creat_list_from_array(self.weights))
        self.biases_len = len(ArrayManipulation().creat_list_from_array(self.biases))
        
        # Armazenando pesos e biases quantizados
        self.weights_quant = self.weights
        self.biases_quant = self.biases

        # Shapes dos pesos e biases
        self.shape_weight = ArrayManipulation().get_shape_from_array(self.weights)
        self.shape_bias = ArrayManipulation().get_shape_from_array(self.biases)

    def __initialize_weight_type(self, shape, init_type):
        if init_type == 'RandomNormal':
            return np.random.randn(*shape)  # Normal padrão
        elif init_type == 'RandomUniform':
            return np.random.uniform(-1, 1, size=shape)  # Uniforme entre -1 e 1
        elif init_type == 'GlorotUniform':
            limit = np.sqrt(6 / (shape[0] + shape[1]))  # Xavier (Glorot)
            return np.random.uniform(-limit, limit, size=shape)
        elif init_type == 'HeNormal':
            return np.random.randn(*shape) * np.sqrt(2 / shape[0])  # He Normal
        else:
            raise ValueError(f"Unrecognized initializer {init_type}.")
        
    
    def __initialize_biases(self, layer_sizes):
        """
        Initialises the weights and biases of the neural network with the specified initialisation method.
        
        Args:
            layer_sizes (list): List with the number of units in each layer.       
        Returns:
            biases (list): List with the initialised biases for each layer.
        """
        try:
            biases = [
                    np.random.randn(1, layer_sizes[i + 1]) * 0.01  # Biases sempre pequenos
                    for i in range(len(layer_sizes) - 1)
                ]
            return biases
        except Exception as e:
            print(f"An error occurred when initialising the biases ({e}). ")


    def __initialize_weights(self, layer_sizes, weight_bias_init):
        """
        Initialises the weights and biases of the neural network with the specified initialisation method.
        
        Args:
            layer_sizes (list): List with the number of units in each layer.
            weight_bias_init (str): Initialisation method for the weights.
        
        Returns:
            weights (list): List with the initialised weights for each layer.
        """
        try:
            weights = [
                    self.__initialize_weight_type((layer_sizes[i], layer_sizes[i + 1]), weight_bias_init)
                    for i in range(len(layer_sizes) - 1)
                ]
            return weights
        except Exception as e:
            print(f"An error occurred when initialising the weights biases ({e}). ")

        
    def __sigmoid(self, x):
        try:
            # Clip the input to avoid overflow in the exponential function
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the sigmoid function: {e}")

    
    def __sigmoid_derivative(self, x):
        try:
            return x * (1 - x)
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the sigmoid derivative: {e}")

    def __relu(self, x):
        try:
            return np.maximum(0, x)
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the ReLU function: {e}")

    def __relu_derivative(self, x):
        try:
            return np.where(x > 0, 1, 0)
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the ReLU derivative: {e}")

    def __leaky_relu(self, x, alpha=0.01):
        try:
            return np.where(x > 0, x, alpha * x)
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the Leaky ReLU function: {e}")

    def __leaky_relu_derivative(self, x, alpha=0.01):
        try:
            return np.where(x > 0, 1, alpha)
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the Leaky ReLU derivative: {e}")

    def __tanh(self, x):
        try:
            return np.tanh(x)
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the tanh function: {e}")

    def __tanh_derivative(self, x):
        try:
            return 1 - np.tanh(x) ** 2
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the tanh derivative: {e}")

    def __elu(self, x, alpha=1.0):
        try:
            return np.where(x > 0, x, alpha * (np.exp(x) - 1))
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the ELU function: {e}")

    def __elu_derivative(self, x, alpha=1.0):
        try:
            return np.where(x > 0, 1, alpha * np.exp(x))
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the ELU derivative: {e}")

    def __softmax(self, x):
        try:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            exps = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the softmax function: {e}")

    def __softmax_derivative(self, output):
        try:
            return np.diag(output) - np.outer(output, output)
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the softmax derivative: {e}")

    def __softplus(self, x):
        try:
            return np.log1p(np.exp(x))  # log(1 + exp(x))
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the softplus function: {e}")

    def __softplus_derivative(self, x):
        try:
            return 1 / (1 + np.exp(-x))
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the softplus derivative: {e}")

    def __swish(self, x):
        try:
            return x * self.__sigmoid(x)
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the swish function: {e}")

    def __swish_derivative(self, x):
        try:
            sig = self.__sigmoid(x)
            return sig + x * sig * (1 - sig)
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the swish derivative: {e}")

    
    def __activate(self, x, activation):
        """
        Applies the specified activation function to the input.

        This private method dynamically evaluates and applies the given activation function
        to the input `x`, supporting a range of commonly used functions in neural networks.

        Parameters:
            x (array-like): The input data to which the activation function will be applied.
            activation (str): The name of the activation function to apply. Supported values include:
                - 'sigmoid': Applies the sigmoid activation function.
                - 'relu': Applies the Rectified Linear Unit (ReLU) activation function.
                - 'leaky_relu': Applies the Leaky ReLU activation function.
                - 'tanh': Applies the hyperbolic tangent (tanh) activation function.
                - 'elu': Applies the Exponential Linear Unit (ELU) activation function.
                - 'softmax': Applies the softmax activation function for probabilistic outputs.
                - 'softplus': Applies the softplus activation function.
                - 'swish': Applies the swish activation function.
                - 'linear': Returns the input `x` unchanged (linear activation).

        Returns:
            Transformed input `x` after applying the specified activation function.

        Raises:
            ValueError: If the provided `activation` is not a supported function name.
        """
        if activation == 'sigmoid':
            return self.__sigmoid(x)
        elif activation == 'relu':
            return self.__relu(x)
        elif activation == 'leaky_relu':
            return self.__leaky_relu(x)
        elif activation == 'tanh':
            return self.__tanh(x)
        elif activation == 'elu':
            return self.__elu(x)
        elif activation == 'softmax':
            return self.__softmax(x)
        elif activation == 'softplus':
            return self.__softplus(x)
        elif activation == 'swish':
            return self.__swish(x)
        elif activation == 'linear':
            return x  
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def __activate_derivative(self, x, activation):
        """
        Computes the derivative of the specified activation function for the given input.

        This private method dynamically evaluates and applies the derivative of the given
        activation function to the input `x`. It supports a range of commonly used functions
        in neural networks.

        Parameters:
            x (array-like): The input data for which the derivative of the activation 
                function will be computed.
            activation (str): The name of the activation function whose derivative to apply. 
                Supported values include:
                - 'sigmoid': Computes the derivative of the sigmoid function.
                - 'relu': Computes the derivative of the Rectified Linear Unit (ReLU).
                - 'leaky_relu': Computes the derivative of the Leaky ReLU function.
                - 'tanh': Computes the derivative of the hyperbolic tangent (tanh) function.
                - 'elu': Computes the derivative of the Exponential Linear Unit (ELU).
                - 'softmax': Computes the derivative of the softmax function.
                - 'softplus': Computes the derivative of the softplus function.
                - 'swish': Computes the derivative of the swish function.
                - 'linear': Returns 1 (derivative of the linear activation function).

        Returns:
            The computed derivative of the specified activation function applied to `x`.

        Raises:
            ValueError: If the provided `activation` is not a supported function name.
        """
        if activation == 'sigmoid':
            return self.__sigmoid_derivative(x)
        elif activation == 'relu':
            return self.__relu_derivative(x)
        elif activation == 'leaky_relu':
            return self.__leaky_relu_derivative(x)
        elif activation == 'tanh':
            return self.__tanh_derivative(x)
        elif activation == 'elu':
            return self.__elu_derivative(x)
        elif activation == 'softmax':
            return self.__softmax_derivative(x)  
        elif activation == 'softplus':
            return self.__softplus_derivative(x)
        elif activation == 'swish':
            return self.__swish_derivative(x)
        elif activation == 'linear':
            return 1  
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    



    def __mean_squared_error_loss(self, y_true, y_pred):
        """
        Computes the Mean Squared Error (MSE) loss.

        Parameters:
            y_true (array-like): Ground truth values.
            y_pred (array-like): Predicted values.

        Returns:
            float: The mean squared error loss.

        Raises:
            RuntimeError: If an error occurs during the computation.
        """
        try:
            return np.mean(np.square(y_true - y_pred))
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the Mean Squared Error loss: {e}")

    def __mean_absolute_error_loss(self, y_true, y_pred):
        """
        Computes the Mean Absolute Error (MAE) loss.

        Parameters:
            y_true (array-like): Ground truth values.
            y_pred (array-like): Predicted values.

        Returns:
            float: The mean absolute error loss.

        Raises:
            RuntimeError: If an error occurs during the computation.
        """
        try:
            return np.mean(np.abs(y_true - y_pred))
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the Mean Absolute Error loss: {e}")

    def __cross_entropy_loss(self, y_true, y_pred):
        """
        Computes the Cross-Entropy loss for classification problems.

        Parameters:
            y_true (array-like): Ground truth probabilities or one-hot encoded labels.
            y_pred (array-like): Predicted probabilities.

        Returns:
            float: The cross-entropy loss.

        Raises:
            RuntimeError: If an error occurs during the computation.
        """
        try:
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid numerical instability
            return -np.mean(y_true * np.log(y_pred))
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the Cross-Entropy loss: {e}")

    def __binary_cross_entropy_loss(self, y_true, y_pred):
        """
        Computes the Binary Cross-Entropy (BCE) loss for binary classification problems.

        Parameters:
            y_true (array-like): Ground truth binary labels (0 or 1).
            y_pred (array-like): Predicted probabilities.

        Returns:
            float: The binary cross-entropy loss.

        Raises:
            RuntimeError: If an error occurs during the computation.
        """
        try:
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid numerical instability
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        except Exception as e:
            raise RuntimeError(f"An error occurred while computing the Binary Cross-Entropy loss: {e}")




    def __sgd_optimizer(self, gradient, learning_rate):
        """
        Computes the update for the Stochastic Gradient Descent (SGD) optimizer.

        Parameters:
            gradient (array-like): The gradient of the loss function with respect to the parameters.
            learning_rate (float): The learning rate for the optimizer.

        Returns:
            array-like: The update to be applied to the parameters.

        Raises:
            RuntimeError: If an error occurs during the computation.
        """
        try:
            return learning_rate * gradient
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the SGD optimizer: {e}")

    def __adam_optimizer(self, gradient, learning_rate, beta1, beta2, epsilon, m=None, v=None, t=0):
        """
        Computes the update for the Adam optimizer, an adaptive learning rate optimization algorithm.

        Parameters:
            gradient (array-like): The gradient of the loss function with respect to the parameters.
            learning_rate (float): The learning rate for the optimizer.
            beta1 (float): Exponential decay rate for the first moment estimate.
            beta2 (float): Exponential decay rate for the second moment estimate.
            epsilon (float): A small value to avoid division by zero.
            m (array-like, optional): The first moment estimate. Defaults to None.
            v (array-like, optional): The second moment estimate. Defaults to None.
            t (int, optional): The time step. Defaults to 0.

        Returns:
            tuple: A tuple containing the update, m, v, and t. The update is the value to be applied to the parameters,
                m and v are the updated moment estimates, and t is the updated time step.

        Raises:
            RuntimeError: If an error occurs during the computation.
        """
        try:
            if m is None:
                m = np.zeros_like(gradient)
                v = np.zeros_like(gradient)
            
            t += 1
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            return update, m, v, t
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the Adam optimizer: {e}")

    def __adamax_optimizer(self, gradient, learning_rate, beta1, beta2, epsilon, m=None, u=None, t=0):
        """
        Computes the update for the AdaMax optimizer, a variant of the Adam optimizer.

        Parameters:
            gradient (array-like): The gradient of the loss function with respect to the parameters.
            learning_rate (float): The learning rate for the optimizer.
            beta1 (float): Exponential decay rate for the first moment estimate.
            beta2 (float): Exponential decay rate for the second moment estimate.
            epsilon (float): A small value to avoid division by zero.
            m (array-like, optional): The first moment estimate. Defaults to None.
            u (array-like, optional): The maximum of the second moment estimate. Defaults to None.
            t (int, optional): The time step. Defaults to 0.

        Returns:
            tuple: A tuple containing the update, m, u, and t. The update is the value to be applied to the parameters,
                m is the updated first moment estimate, u is the updated second moment estimate, and t is the updated time step.

        Raises:
            RuntimeError: If an error occurs during the computation.
        """
        try:
            if m is None:
                m = np.zeros_like(gradient)
                u = np.zeros_like(gradient)
            
            t += 1
            m = beta1 * m + (1 - beta1) * gradient
            u = np.maximum(beta2 * u, np.abs(gradient))
            update = learning_rate * m / (u + epsilon)
            return update, m, u, t
        except Exception as e:
            raise RuntimeError(f"An error occurred while applying the AdaMax optimizer: {e}")


    def predict(self, input: list):
        """
        Makes a prediction based on the input `input` using the model's weights and biases.

        Parameters:
            input (array-like): The input data to be predicted. This can be a single instance or a batch of data.
        Returns:
            array-like: The final output after passing through all layers and activation functions.

        Raises:
            ValueError: If the input data `input` does not match the expected shape or type.
            RuntimeError: If an error occurs during the prediction process (e.g., issues with the weights, biases, or activation functions).
        """
        try:
            result = [input]
            if not self.training_with_quantization:
                # Predict using regular weights and biases
                for i in range(len(self.weights)):
                    z = np.dot(result[-1], self.weights[i]) + self.biases[i]
                    activation = self.__activate(z, self.activation_functions[i])
                    result.append(activation)
                return result[-1]
            else:
                # Predict using quantized weights and biases
                for i in range(len(self.weights_quant)):
                    z = np.dot(result[-1], self.weights_quant[i]) + self.biases_quant[i]
                    activation = self.__activate(z, self.activation_functions[i])
                    result.append(activation)
                return result[-1]
        except ValueError as e:
            raise ValueError(f"Invalid input data shape: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during the prediction process: {e}")



    def __forward_quant(self, x):
        """
        Performs a forward pass of the quantized neural network, processing input `x` through the quantized weights and biases.

        Parameters:
            x (array-like): The input data to be processed through the quantized network layers.

        Returns:
            array-like: The final output after passing through all quantized layers and activation functions.

        Raises:
            ValueError: If the input data `x` does not match the expected shape or type.
            RuntimeError: If an error occurs during the forward pass (e.g., issues with the quantized weights, biases, or activation functions).
        """
        try:
            self.activations = [x]
            for i in range(len(self.weights_quant)):
                z = np.dot(self.activations[-1], self.weights_quant[i]) + self.biases_quant[i]
                activation = self.__activate(z, self.activation_functions[i])
                self.activations.append(activation)
            return self.activations[-1]
        except ValueError as e:
            raise ValueError(f"Invalid input data shape: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during the forward pass with quantized weights: {e}")


    def __forward(self, x):
        """
        Performs a forward pass of the neural network, processing input `x` through the layers with standard weights and biases.

        Parameters:
            x (array-like): The input data to be processed through the network layers.

        Returns:
            array-like: The final output after passing through all layers and activation functions.

        Raises:
            ValueError: If the input data `x` does not match the expected shape or type.
            RuntimeError: If an error occurs during the forward pass (e.g., issues with the weights, biases, or activation functions).
        """
        try:
            self.activations = [x]
            for i in range(len(self.weights)):
                z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
                activation = self.__activate(z, self.activation_functions[i])
                self.activations.append(activation)
            return self.activations[-1]
        except ValueError as e:
            raise ValueError(f"Invalid input data shape: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during the forward pass with standard weights: {e}")



    def __select_loss_function(self, loss_function:str):
        """
        Selects the appropriate loss function based on the given `loss_function` argument.

        Parameters:
            loss_function (str): The name of the desired loss function. Supported options are:
                                'mean_squared_error', 'cross_entropy', 'mean_absolute_error', and 'binary_cross_entropy'.

        Returns:
            function: The selected loss function that can be used for training the model.

        Raises:
            ValueError: If the specified loss function is not recognized.

        Example:
            loss_fn = model.select_loss_function('mean_squared_error')
            loss_value = loss_fn(y_true, y_pred)
        """
        if loss_function == 'mean_squared_error':
            loss_fn = self.__mean_squared_error_loss
        elif loss_function == 'cross_entropy':
            loss_fn = self.__cross_entropy_loss
        elif loss_function == 'mean_absolute_error':
            loss_fn = self.__mean_absolute_error_loss
        elif loss_function == 'binary_cross_entropy':
            loss_fn = self.__binary_cross_entropy_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        return loss_fn


    def __select_optimizer(self, optimizer):
        """
        Selects the appropriate optimizer based on the given `optimizer` argument.

        Parameters:
            optimizer (str): The name of the desired optimizer. Supported options are:
                            'sgd', 'adam', and 'adamax'.

        Returns:
            function: The selected optimizer function that can be used for updating model parameters.

        Raises:
            ValueError: If the specified optimizer is not recognized.

        Example:
            optimizer_fn = model.select_optimizer('adam')
            updated_params = optimizer_fn(gradient, learning_rate, beta1, beta2, epsilon)
        """
        if optimizer == 'sgd':
            optimizer_fn = self.__sgd_optimizer
        elif optimizer == 'adam':
            optimizer_fn = self.__adam_optimizer
        elif optimizer == 'adamax':
            optimizer_fn = self.__adamax_optimizer
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        return optimizer_fn



    def __backward(self, y, output):
        """
        Performs the backpropagation step to adjust the neural network's weights.

        Parameters:
            y (array): Expected output (true labels).
            output (array): Current output of the network for the input data.

        Returns:
            gradients (list): List of calculated gradients to update the weights.
        """
        try:
            # Compute the initial error at the output (difference between expected and actual output)
            errors = [y - output]

            # Calculate the delta for the output layer using the derivative of the activation function
            deltas = [errors[-1] * self.__activate_derivative(output, self.activation_functions[-1])]

            # Iterate over the hidden layers to calculate deltas
            for i in range(len(self.weights)-1, 0, -1):
                error = deltas[-1].dot(self.weights[i].T)  # Propagate the error to the previous layer
                delta = error * self.__activate_derivative(self.activations[i], self.activation_functions[i])
                deltas.append(delta)

            # Reverse deltas to align with the layer order
            deltas.reverse()

            # Calculate gradients for weights using deltas and activations
            gradients = []
            for i in range(len(self.weights)):
                gradient = self.activations[i].T.dot(deltas[i])  # Gradient for the current layer
                gradients.append(gradient)
            
            return gradients

        except IndexError as e:
            # Common error: index out of range, possibly due to mismatched weights or activations
            raise ValueError("Check the alignment of weights, activations, and activation functions.") from e
        except AttributeError as e:
            # Common error: missing attribute, possibly caused by incorrect initialization
            raise AttributeError("Ensure all necessary attributes, such as weights and activation functions, are properly initialized.") from e
        except Exception as e:
            # General handler for unexpected errors
            raise RuntimeError("An unexpected error occurred during backpropagation.") from e

    

    def __backward_quant(self, y, output):
        """
        Performs the backpropagation step to adjust the quantized neural network weights.

        Parameters:
            y (array): Expected output (true labels).
            output (array): Current network output for the input data.

        Returns:
            gradients (list): List of computed gradients for weight adjustment.
        """
        try:
            # Compute the initial error at the output layer (difference between expected and actual output)
            errors = [y - output]

            # Calculate the output layer delta using the activation function's derivative
            deltas = [errors[-1] * self.__activate_derivative(output, self.activation_functions[-1])]

            # Iterate through hidden layers to compute deltas (using quantized weights)
            for i in range(len(self.weights_quant)-1, 0, -1):
                error = deltas[-1].dot(self.weights_quant[i].T)  # Propagate the error to the previous layer using quantized weights
                delta = error * self.__activate_derivative(self.activations[i], self.activation_functions[i])
                deltas.append(delta)

            # Reverse the deltas to align with the layer order
            deltas.reverse()

            # Compute gradients for the weights using deltas and activations
            gradients = []
            for i in range(len(self.weights)):
                gradient = self.activations[i].T.dot(deltas[i])  # Gradient for the current layer
                gradients.append(gradient)
            
            return gradients

        except IndexError as e:
            # Common error: index out of range, possibly due to mismatched quantized weights or activations
            raise ValueError("Check the alignment of quantized weights, activations, and activation functions.") from e
        except AttributeError as e:
            # Common error: missing attribute, possibly caused by improper initialization
            raise AttributeError("Ensure all necessary variables like quantized weights and activation functions are properly initialized.") from e
        except Exception as e:
            # Generic handling for other unexpected errors
            raise RuntimeError("Unexpected error occurred during backpropagation with quantized weights.") from e


    def train(self,
            X: list,
            y: list,
            epochs: int = 100,
            learning_rate: float = 0.001,
            loss_function: str = 'mean_absolute_error',
            optimizer: str = 'adam',
            batch_size: int = 36,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 1e-7,
            epochs_quantization: int = 50,
            distance_metric: str = "euclidean",
            bias_clustering_method=None,
            weight_clustering_method=None,
            validation_split: float = 0.2):
        """
        Train the model using the provided input and target data, with validation.

        Args:
            input (list): Training input data.
            target (list): Training target data.
            epochs (int): Number of training epochs (default: 100).
            learning_rate (float): Learning rate for parameter updates (default: 0.001).
            loss_function (str): Loss function to optimize (default: 'mean_absolute_error').
            optimizer (str): Optimization algorithm to use (default: 'adam').
            batch_size (int): Batch size for mini-batch training (default: 36).
            beta1 (float): Exponential decay rate for the first moment in Adam (default: 0.9).
            beta2 (float): Exponential decay rate for the second moment in Adam (default: 0.999).
            epsilon (float): Small constant to avoid division by zero in Adam (default: 1e-7).
            epochs_quantization (int): Number of epochs to apply quantization (default: 50).
            distance_metric (str): Distance metric for clustering (default: "euclidean").
            bias_clustering_method: Method used for clustering biases (default: None).
            weight_clustering_method: Method used for clustering weights (default: None).
            validation_split (float): Proportion of data to use for validation (default: 0.2).

        Raises:
            ValueError: If input or target are empty, or if parameters have invalid configurations.
            RuntimeError: For unexpected errors during training or validation processes.
        """
        try:
            # Validate inputs
            if X.size == 0 or y.size == 0:
                raise ValueError("Input and target data cannot be empty.")
            if len(X) != len(y):
                raise ValueError("Input and target must have the same length.")
            if not (0 < validation_split < 1):
                raise ValueError("Validation split must be a float between 0 and 1.")
            
            # Split the data into training and validation sets
            validation_size = int(len(X) * validation_split)
            train_size = len(X) - validation_size

            input_train, input_val = X[:train_size], X[train_size:]
            target_train, target_val = y[:train_size], y[train_size:]

            if len(input_train) == 0 or len(input_val) == 0:
                raise ValueError("Validation split resulted in empty training or validation sets.")
           

            # Select the loss function and optimizer
            loss_fn = self.__select_loss_function(loss_function=loss_function)
            optimizer_fn = self.__select_optimizer(optimizer=optimizer)

            # Default batch size if not provided
            if batch_size is None:
                batch_size = len(input_train)

            # Training loop
            for epoch in range(epochs):
                try:
                    # Shuffle the training dataset at the beginning of each epoch
                    indices = np.random.permutation(len(input_train))
                    X_shuffled = input_train[indices]
                    y_shuffled = target_train[indices]

                    # Quantization process
                    if (epoch > (epochs - epochs_quantization)) and self.training_with_quantization:
                        try:
                            # Convert weights and biases into lists for clustering
                            weight_list = ArrayManipulation().creat_list_from_array(self.weights)
                            bias_list = ArrayManipulation().creat_list_from_array(self.biases)
                            
                            # Apply clustering to weights and biases
                            self.bias_center = self.applying_clusterings(
                                clustering_method=bias_clustering_method, parameter_list=bias_list)
                            self.weight_center = self.applying_clusterings(
                                clustering_method=weight_clustering_method, parameter_list=weight_list)
                            
                            # Find closest cluster centers
                            result_df_bias = self.find_closest_values(
                                bias_list, self.bias_center, distance_metric=distance_metric)
                            result_df_weight = self.find_closest_values(
                                weight_list, self.weight_center, distance_metric=distance_metric)
                            
                            # Convert clustered results back into arrays
                            self.weights_quant = ArrayManipulation().creat_array_from_list(result_df_weight['Center'].values, self.shape_weight)
                            self.biases_quant = ArrayManipulation().creat_array_from_list(result_df_bias['Center'].values, self.shape_bias)
                            
                            # Store indices for clustered values
                            self.weights_index = ArrayManipulation().creat_array_from_list(result_df_weight['Index'].values, self.shape_weight)
                            self.biases_index = ArrayManipulation().creat_array_from_list(result_df_bias['Index'].values, self.shape_bias)
                        except Exception as e:
                            raise RuntimeError(f"Error during quantization clustering: {str(e)}")
                    
                    # Mini-batch training
                    for i in range(0, len(input_train), batch_size):
                        X_batch = X_shuffled[i:i+batch_size]
                        y_batch = y_shuffled[i:i+batch_size]
                        try:
                            if (epoch >= (epochs - epochs_quantization)) and self.training_with_quantization:
                                # Forward pass with quantization
                                output = self.__forward_quant(X_batch)
                                loss = loss_fn(y_batch, output)
                                gradient = self.__backward_quant(y_batch, output)
                                # Update parameters using quantization-aware method
                                self.__update_params(gradient, learning_rate, optimizer_fn, beta1, beta2, epsilon)
                            else:
                                # Standard forward pass
                                output = self.__forward(X_batch)
                                loss = loss_fn(y_batch, output)
                                gradient = self.__backward(y_batch, output)
                                # Standard parameter update
                                self.__update_params(gradient, learning_rate, optimizer_fn, beta1, beta2, epsilon)
                        except Exception as e:
                            raise RuntimeError(f"Error during mini-batch training at epoch {epoch}, batch {i}: {str(e)}")
                    
                    # Validation at the end of each epoch
                    val_output = self.__forward(input_val)
                    val_loss = loss_fn(target_val, val_output)

                    # Log epoch results
                    if (epoch > (epochs - epochs_quantization)) and self.training_with_quantization:
                        print(f"Epoch {epoch}/{epochs}, Loss: {loss}, Val Loss: {val_loss}, Bias Clusters: {len(self.bias_center)}, Weight Clusters: {len(self.weight_center)}")
                        self.loss_cumulative.append(loss)
                        self.bias_cluster_cumulative.append(len(self.bias_center))
                        self.weight_cluster_cumulative.append(len(self.weight_center))
                    else:
                        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}, Val Loss: {val_loss}")
                        self.loss_cumulative.append(loss)
                        self.validation_loss_cumulative.append(val_loss)
                except Exception as e:
                    raise RuntimeError(f"Error during epoch {epoch}: {str(e)}")
        except Exception as e:
            # General exception handling
            raise RuntimeError(f"An error occurred during training: {str(e)}")


                
    def __update_params(self, gradients, learning_rate, optimizer_fn, beta1=None, beta2=None, epsilon=1e-8, previous_updates=None, ms=None, vs=None, t=0):
        """
        Update model parameters (weights and biases) using the specified optimizer.

        Args:
            gradients (list): Gradients computed during backpropagation for each layer.
            learning_rate (float): Learning rate for parameter updates.
            optimizer_fn (function): Optimizer function to use (e.g., SGD, Adam).
            beta1 (float, optional): Exponential decay rate for the first moment estimate (Adam). Default is None.
            beta2 (float, optional): Exponential decay rate for the second moment estimate (Adam). Default is None.
            epsilon (float): Small constant to avoid division by zero (default: 1e-8).
            previous_updates (list, optional): Previous updates for each parameter (used by some optimizers like Momentum). Default is None.
            ms (list, optional): First moment estimates (used by Adam). Default is None.
            vs (list, optional): Second moment estimates (used by Adam). Default is None.
            t (int, optional): Time step for the optimizer (used by Adam). Default is 0.

        Raises:
            ValueError: If gradients, weights, or biases are missing or improperly configured.
            TypeError: If optimizer_fn is not callable.
            RuntimeError: If an error occurs during the parameter update process.

        """
        try:
            # Validate inputs
            if not self.weights or not self.biases:
                raise ValueError("Model weights or biases are not initialized.")
            if not gradients or len(gradients) != len(self.weights):
                raise ValueError("Gradients must match the number of model weights.")
            if not callable(optimizer_fn):
                raise TypeError("The optimizer_fn must be a callable function.")
            if learning_rate <= 0:
                raise ValueError("Learning rate must be positive.")

            # Initialize variables if not provided
            if previous_updates is None:
                previous_updates = [None] * len(gradients)
            if ms is None:
                ms = [None] * len(gradients)
                vs = [None] * len(gradients)

            # Iterate through all layers to update weights and biases
            for i in range(len(self.weights)):
                try:
                    # Check if the optimizer is SGD (simpler case)
                    if optimizer_fn == self.__sgd_optimizer:
                        update = optimizer_fn(gradients[i], learning_rate)
                    else:
                        # Optimizers like Adam require additional parameters (ms, vs, t)
                        update, m, v, t = optimizer_fn(
                            gradients[i], learning_rate, beta1, beta2, epsilon, ms[i], vs[i], t
                        )
                        ms[i], vs[i] = m, v  # Update moment estimates

                    # Update parameters
                    previous_updates[i] = update
                    self.weights[i] += update
                    self.biases[i] += np.sum(update, axis=0, keepdims=True) * learning_rate

                except Exception as e:
                    # Catch and report any issues during parameter update for a specific layer
                    raise RuntimeError(f"Error updating parameters for layer {i}: {str(e)}")

        except Exception as e:
            # General exception handling for unexpected errors
            raise RuntimeError(f"An error occurred during parameter updates: {str(e)}")



    def applying_clusterings(self, clustering_method, parameter_list):
        """
        Applies the specified clustering method to a given parameter list and returns the computed centers.

        Args:
            clustering_method (tuple): A tuple containing the method name (str) and the clustering object.
            parameter_list (list): A list of parameters to be clustered.

        Returns:
            list: A list of cluster centers computed by the specified method.

        Raises:
            ValueError: If the clustering method is unknown or invalid.
        """
        method_name, clusterization = clustering_method
        center_list = []

        try:
            if method_name == "AUTOCLOUD":
                self._apply_autocloud(clusterization, parameter_list, center_list)

            elif method_name in {"MEANSHIFT", "AFFINITYPROP"}:
                self._apply_batch_clustering(clusterization, parameter_list, center_list)

            elif method_name == "DBSTREAM":
                self._apply_dbstream(clusterization, parameter_list, center_list)

            else:
                raise ValueError(f"Unknown clustering method: {method_name}")

            return center_list

        except Exception as e:
            raise RuntimeError(f"Error applying clustering method {method_name}: {e}")


    def _apply_autocloud(self, clusterization, parameter_list, center_list):
        """
        Helper method for applying the AutoCloud clustering method.
        """
        for bias in parameter_list:
            clusterization.run(bias)
        for cloud in clusterization.c:
            center_list.append(cloud.mean.tolist()[0])


    def _apply_batch_clustering(self, clusterization, parameter_list, center_list):
        """
        Helper method for batch-based clustering methods (MeanShift and Affinity Propagation).
        """
        if len(parameter_list) > 150:
            chunk_size = 100
            chunks = [parameter_list[i:i + chunk_size] for i in range(0, len(parameter_list), chunk_size)]
            for chunk in chunks:
                clusterization.fit(chunk)
        else:
            clusterization.fit(parameter_list)
        center_list.extend(clusterization.cluster_centers_.reshape(1, -1)[0].tolist())


    def _apply_dbstream(self, clusterization, parameter_list, center_list):
        """
        Helper method for applying the DBSTREAM clustering method.
        """
        for x, _ in stream.iter_array(parameter_list):
            clusterization.learn_one(x=x)
        for i in range(clusterization.n_clusters):
            center_list.append(clusterization.centers[i][0])



    @staticmethod
    def __mahalanobis_distance(a, b, data):
        # Garantir que a e b sejam arrays 1D
        a = np.array(a)
        b = np.array(b)

        # Calcular o desvio padrão dos dados
        std_dev = np.std(data)

        # Se o desvio padrão for zero, não é possível calcular a Mahalanobis
        if std_dev == 0:
            raise ValueError("O desvio padrão é zero, impossível calcular a distância de Mahalanobis.")

        # Calcular a diferença entre os pontos
        delta = a - b

        # Calcular a distância de Mahalanobis
        dist = np.abs(delta) / std_dev

        return dist



    @staticmethod
    def __euclidean_distance(a, b):
        if isinstance(a, (int, float)):
            a = [a]
        if isinstance(b, (int, float)):
            b = [b]
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

    @staticmethod
    def __manhattan_distance(a, b):
        if isinstance(a, (int, float)):
            a = [a]
        if isinstance(b, (int, float)):
            b = [b]
        return sum(abs(ai - bi) for ai, bi in zip(a, b))

    @staticmethod
    def __minkowski_distance(a, b, p=3):
        if isinstance(a, (int, float)):
            a = [a]
        if isinstance(b, (int, float)):
            b = [b]
        return sum(abs(ai - bi) ** p for ai, bi in zip(a, b)) ** (1 / p)

    @staticmethod
    def __chebyshev_distance(a, b):
        if isinstance(a, (int, float)):
            a = [a]
        if isinstance(b, (int, float)):
            b = [b]
        return max(abs(ai - bi) for ai, bi in zip(a, b))

    @staticmethod
    def __cosine_distance(a, b):
        if isinstance(a, (int, float)):
            a = [a]
        if isinstance(b, (int, float)):
            b = [b]
        dot_product = sum(ai * bi for ai, bi in zip(a, b))
        magnitude_a = sum(ai ** 2 for ai in a) ** 0.5
        magnitude_b = sum(bi ** 2 for bi in b) ** 0.5
        return 1 - (dot_product / (magnitude_a * magnitude_b))

    @staticmethod
    def __hamming_distance(a, b):
        if isinstance(a, (int, float)):
            a = [a]
        if isinstance(b, (int, float)):
            b = [b]
        return sum(1 for ai, bi in zip(a, b) if ai != bi)

    @staticmethod
    def __bray_curtis_distance(a, b):
        if isinstance(a, (int, float)):
            a = [a]
        if isinstance(b, (int, float)):
            b = [b]
        numerator = sum(abs(ai - bi) for ai, bi in zip(a, b))
        denominator = sum(abs(ai + bi) for ai, bi in zip(a, b))
        return numerator / denominator

    @staticmethod
    def __jaccard_distance(a, b):
        if isinstance(a, (int, float)):
            a = [a]
        if isinstance(b, (int, float)):
            b = [b]
        set_a, set_b = set(a), set(b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return 1 - intersection / union

    @staticmethod
    def __wasserstein_distance(a, b):
        if isinstance(a, (int, float)):
            a = [a]
        if isinstance(b, (int, float)):
            b = [b]
        sorted_a, sorted_b = sorted(a), sorted(b)
        return sum(abs(ai - bi) for ai, bi in zip(sorted_a, sorted_b))

    @staticmethod
    def __dtw_distance(a, b):
        if isinstance(a, (int, float)):
            a = [a]
        if isinstance(b, (int, float)):
            b = [b]
        n, m = len(a), len(b)
        dtw_matrix = [[float('inf')] * m for _ in range(n)]
        dtw_matrix[0][0] = abs(a[0] - b[0])

        for i in range(n):
            for j in range(m):
                cost = abs(a[i] - b[j])
                if i > 0:
                    dtw_matrix[i][j] = min(dtw_matrix[i][j], dtw_matrix[i-1][j] + cost)
                if j > 0:
                    dtw_matrix[i][j] = min(dtw_matrix[i][j], dtw_matrix[i][j-1] + cost)
                if i > 0 and j > 0:
                    dtw_matrix[i][j] = min(dtw_matrix[i][j], dtw_matrix[i-1][j-1] + cost)

        return dtw_matrix[-1][-1]

    def find_closest_values(self, array_parameter:list, array_centers:list, distance_metric:str = "euclidean"):
        """
        Finds the closest centers for each value in the array_parameter based on the specified distance type.

        Args:
            array_parameter (list): List of values for which the closest centers are to be found.
            array_centers (list): List of reference centers.
            distance_metric (str): Type of distance metric to use, e.g., "euclidean", "manhattan".

        Returns:
            pd.DataFrame: A DataFrame containing the original value, the closest center, and the center index.

        Raises:
            ValueError: If the specified distance type is not recognized.
            TypeError: If the input arrays are not iterable.
        """
        try:
            # Basic validation to ensure input parameters are iterable.
            if not hasattr(array_parameter, "__iter__") or not hasattr(array_centers, "__iter__"):
                raise TypeError("The parameters array_parameter and array_centers must be iterable.")
            
            # Validation for recognized distance types.
            available_distances = [
                "euclidean", "manhattan", "minkowski", "chebyshev", 
                "cosine", "hamming", "bray_curtis", "jaccard", 
                "wasserstein", "dtw", "mahalanobis"
            ]
            if distance_metric not in available_distances:
                raise ValueError(f"Unknown distance type: {distance_metric}. Valid options are: {', '.join(available_distances)}")

            results = []

            # Iterate over each value in the input array.
            for value in array_parameter:
                distances = []  # List to store calculated distances for each center.

                # Calculate the distance for each center based on the specified type.
                for center in array_centers:
                    try:
                        if distance_metric == "euclidean":
                            distance = self.__euclidean_distance(value, center)
                        elif distance_metric == "manhattan":
                            distance = self.__manhattan_distance(value, center)
                        elif distance_metric == "minkowski":
                            distance = self.__minkowski_distance(value, center)
                        elif distance_metric == "chebyshev":
                            distance = self.__chebyshev_distance(value, center)
                        elif distance_metric == "cosine":
                            distance = self.__cosine_distance(value, center)
                        elif distance_metric == "hamming":
                            distance = self.__hamming_distance(value, center)
                        elif distance_metric == "bray_curtis":
                            distance = self.__bray_curtis_distance(value, center)
                        elif distance_metric == "jaccard":
                            distance = self.__jaccard_distance(value, center)
                        elif distance_metric == "wasserstein":
                            distance = self.__wasserstein_distance(value, center)
                        elif distance_metric == "dtw":
                            distance = self.__dtw_distance(value, center)
                        elif distance_metric == "mahalanobis":
                            distance = self.__mahalanobis_distance(value, center, array_parameter)
                        else:
                            # Unlikely case since the type has already been validated.
                            raise ValueError(f"Unknown distance type: {distance_metric}")
                        
                        distances.append(distance)
                    except Exception as e:
                        # Handle any unexpected error during distance calculation.
                        raise RuntimeError(f"Error calculating distance for center {center} with value {value}: {str(e)}")

                # Find the index of the closest center based on the smallest distance.
                try:
                    closest_center_idx = distances.index(min(distances))
                    closest_center = array_centers[closest_center_idx]
                except ValueError as e:
                    # Unlikely case if the distances list is empty.
                    raise RuntimeError(f"Error finding the closest center: {str(e)}")

                # Store the results in a structured format.
                results.append({
                    'Values': value[0],  # Assuming `value` is a list or array.
                    'Center': closest_center,
                    'Index': closest_center_idx
                })

            # Convert the results to a Pandas DataFrame and return.
            df_result = pd.DataFrame(results)
            return df_result
        
        except Exception as e:
            # General exception handling to capture unexpected errors.
            raise RuntimeError(f"An error occurred while executing the function 'find_closest_values': {str(e)}")



    def save_model_as_cpp(self, file_name: str):
        """
        Saves the model to a .h file with the specified file name.
        
        Parameters:
        file_name (str): The name of the file where the model will be saved.
        
        Exceptions:
        - If an error occurs while extracting the model parameters or generating the C++ code, an exception will be raised with the error message.
        - If the file cannot be opened or written to, an exception will be raised.
        """
        json_model = self.__model_parameter_extraction()
        return CppGeneration().save_model_as_cpp(file_name = file_name, json_model = json_model)


    def save_model_as_json(self, file_name: str):
        """
        Saves the model to a .json file with the specified file name.
        
        Parameters:
        file_name (str): The name of the file where the model will be saved (without the .json extension).
        
        Exceptions:
        - If an error occurs while extracting the model parameters or generating the JSON, an exception will be raised with the error message.
        - If the file cannot be opened or written to, an exception will be raised.
        """
        # Extract model parameters in JSON format
        json_model = self.__model_parameter_extraction()
        JsonHandle().save_model_as_json(file_name = file_name, json_model = json_model)
            


        

    def load_json_model(self, file_name: str):
        """
        Loads the contents of a .json file into a Python dictionary.
        
        Parameters:
        file_name (str): The name of the file (with or without the .json extension).
        
        Returns:
        dict: The contents of the JSON file as a dictionary.
        
        Exceptions:
        - FileNotFoundError: Raised if the file does not exist.
        - json.JSONDecodeError: Raised if the file is not a valid JSON format.
        - IOError: Raised if there's an error reading the file.
        """
        return JsonHandle().load_json_model(file_name = file_name)

    

       

    def __model_parameter_extraction(self, ):
        """
        Extracts the model parameters including weights, biases, and activation functions 
        and returns them in a dictionary format.

        Depending on whether the model is trained with quantization or not, the function 
        behaves differently:

        - If the model is trained with quantization:
            - It includes `bias_center` and `weight_center` in the returned model data.
            - It uses indexed weights (`weights_index`) and biases (`biases_index`) for each layer.
        
        - If the model is not trained with quantization:
            - It directly includes the weights and biases for each layer.
        
        The function performs error checks to ensure that the required attributes (weights, 
        biases, and activation functions) exist and are valid. If any required data is missing 
        or incorrectly formatted, it raises appropriate exceptions (e.g., `ValueError`, `TypeError`, 
        `AttributeError`).

        Returns:
            dict: A dictionary containing the extracted model parameters, including the number 
            of layers, the layers' activation functions, and the associated weights and biases.

        Raises:
            ValueError: If required data (e.g., `bias_center`, `weight_center`, `weights`, or `biases`) 
                        is missing or incorrectly formatted.
            TypeError: If weights or biases are not list-like objects (i.e., they cannot be converted 
                    to lists).
            AttributeError: If any expected attribute is missing in the model (e.g., `weights`, 
                            `biases`, `weights_index`, `biases_index`).
            Exception: If an unexpected error occurs during the extraction process.
        """
        try:

            model_data = {'model_quantized': self.training_with_quantization,
                          'num_layers': len(self.weights), 'layers': [], 
                          }
   


            if self.training_with_quantization:
                # Ensure that the required quantization data is present
                if not hasattr(self, 'bias_center') or not hasattr(self, 'weight_center'):
                    raise ValueError("Missing quantization data: 'bias_center' or 'weight_center'.")
                
                model_data['centers_bias'] = self.bias_center
                model_data['centers_weights'] = self.weight_center

                # Ensure weights_index and biases_index are available
                if not hasattr(self, 'weights_index') or not hasattr(self, 'biases_index'):
                    raise ValueError("Missing quantization weight or bias index data.")

                for i, (weights, biases) in enumerate(zip(self.weights_index, self.biases_index)):
                    # Ensure that the weights and biases have the correct structure
                    if not hasattr(weights, 'tolist') or not hasattr(biases, 'tolist'):
                        raise TypeError(f"Layer {i+1} weights or biases are not list-like objects.")

                    layer_data = {
                        'activation': self.activation_functions[i],
                        'weights': weights.tolist(),
                        'biases': biases.tolist()[0]
                    }
                    model_data['layers'].append(layer_data)
            
            else:
                # Ensure weights and biases exist and are properly structured
                if not hasattr(self, 'weights') or not hasattr(self, 'biases'):
                    raise ValueError("Missing 'weights' or 'biases' in the model.")

                for i, (weights, biases) in enumerate(zip(self.weights, self.biases)):
                    # Ensure that weights and biases have the correct structure
                    if not hasattr(weights, 'tolist') or not hasattr(biases, 'tolist'):
                        raise TypeError(f"Layer {i+1} weights or biases are not list-like objects.")
                    
                    layer_data = {
                        'activation': self.activation_functions[i],
                        'weights': weights.tolist(),
                        'biases': biases[0].tolist()
                    }
                    model_data['layers'].append(layer_data)

            return model_data

        except ValueError as e:
            raise ValueError(f"Error in extracting model parameters: {str(e)}") from e
        except TypeError as e:
            raise TypeError(f"Error in extracting model parameters: {str(e)}") from e
        except AttributeError as e:
            raise AttributeError(f"Missing or invalid attribute in the model: {str(e)}") from e
        except Exception as e:
            raise Exception(f"An unexpected error occurred while extracting model parameters: {str(e)}") from e