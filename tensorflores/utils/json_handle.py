import numpy as np
import json 

class JsonHandle:
    """
    A utility class for handling JSON operations related to neural network models, 
    such as converting TensorFlow models to JSON, applying int8 quantization schemes, 
    and loading/saving models as JSON files.

    This class provides methods to:
    - Convert a TensorFlow model into a structured JSON representation.
    - Modify the JSON to apply int8 quantization for weights and biases.
    - Extract parameters (weights, biases, and shapes) from a JSON model.
    - Save a JSON representation of a model to a file.
    - Load a model from a JSON file.

    It aims to facilitate easy storage and retrieval of model data in JSON format 
    and support quantization-aware tasks.
    """


    def __init__(self, ):
        """
        Initializes the Clustering object. Currently, no specific initialization parameters are required.
        """
        pass



    def int8_scheme_change(self, json_data, weights, biases, min_value, max_value):
        """
        Changes JSON data to use int8 scheme for weights and biases.

        Args:
        - json_data (dict): JSON data to be modified.
        - weights (list of np.ndarray): List of quantized weight arrays.
        - biases (list of np.ndarray): List of quantized bias arrays.
        - min_value (float): Minimum value used for quantization.
        - max_value (float): Maximum value used for quantization.

        Returns:
        - json_data (dict): Modified JSON data with int8 scheme applied.
        """
        json_data['model_quantized'] = 'int8'
        json_data['max_value'] = max_value
        json_data['min_value'] = min_value

        try:
            for i in range(len(json_data['layers'])):
                json_data['layers'][i]['weights'] = weights[i].tolist()
                json_data['layers'][i]['biases'] = biases[i].tolist()

        except (IndexError, KeyError) as e:
            raise ValueError(f"Error modifying JSON data: {e}")

        return json_data
    

    def paramters_from_json(self, json_data):
        """
        Extracts weights, biases, and their shapes from JSON data.

        Args:
        - json_data (dict): JSON data containing 'layers' with 'weights' and 'biases'.

        Returns:
        - weights (list of np.ndarray): List of weight arrays for each layer.
        - weights_shape (list of tuple): List of weight array shapes.
        - biases (list of np.ndarray): List of bias arrays for each layer.
        - biases_shape (list of tuple): List of bias array shapes.
        """
        weights = []
        weights_shape = []
        biases = []
        biases_shape = []

        try:
            for layer in json_data['layers']:
                weights.append(np.array(layer['weights']))
                weights_shape.append(np.array(layer['weights']).shape)
                biases.append(np.array([layer['biases']]))
                biases_shape.append(np.array(layer['biases']).shape)

        except KeyError as e:
            raise ValueError(f"JSON data is missing required key: {e}")

        return weights, weights_shape, biases, biases_shape
    


    def convert_tensorflow_model_to_json(self, tensorflow_model):
        """
        Extracts information from a TensorFlow model and organizes it into a dictionary.
        
        Parameters:
        tensorflow_model: The TensorFlow model to extract information from.

        Returns:
        dict: A dictionary containing the model's information, including quantized weights, biases, and activations.
        """
        try:
            # Initialize the model information dictionary
            model_info = {
                'num_layers': len(tensorflow_model.layers),  # Number of layers in the model
                'model_quantized': False
            }
            layers_info = []  # List to store information about each layer

            # Iterate through each layer in the model
            for layer in tensorflow_model.layers:
                layer_info = {}

                # Extract the activation function if it exists
                layer_info['activation'] = (
                    layer.activation.__name__ if hasattr(layer, 'activation') and layer.activation else None
                )

                # Extract weights and biases if they exist
                weights = layer.get_weights()
                if weights:
                    layer_info['weights'] = weights[0].tolist()
                    layer_info['biases'] = weights[1].tolist()
                else:
                    # If no weights or biases, set them as None
                    layer_info['weights'] = None
                    layer_info['biases'] = None

                layers_info.append(layer_info)  # Append layer info to the list

            # Add layer information to the model dictionary
            model_info['layers'] = layers_info
            return model_info

        except KeyError as e:
            print(f"Key error: {e}")  # Handle key-related errors
        except ValueError as e:
            print(f"Value error: {e}")  # Handle value-related errors
        except Exception as e:
            print(f"Unexpected error: {e}")  # Handle any other unexpected errors


 

    def save_model_as_json(self, file_name: str, json_model):
        """
        Saves the model to a .json file with the specified file name.
        
        Parameters:
        file_name (str): The name of the file where the model will be saved (without the .json extension).
        
        Exceptions:
        - If an error occurs while extracting the model parameters or generating the JSON, an exception will be raised with the error message.
        - If the file cannot be opened or written to, an exception will be raised.
        """
        try:
            
            # Convert the dictionary to a JSON-formatted string
            json_model_str = json.dumps(json_model, indent=4)  # Adiciona indentação para melhor legibilidade

            # Save the extracted model parameters to a .json file
            with open(f'{file_name}.json', 'w') as file:
                file.write(json_model_str)
            
            # Confirm that the model has been saved successfully
            print(f"Model saved successfully as '{file_name}.json'!")
        
        except FileNotFoundError:
            # Error if the file path is invalid or cannot be found
            print(f"Error: The file path '{file_name}.json' could not be found or is invalid.")
            raise
        
        except IOError:
            # Generic I/O error
            print(f"Error: An I/O error occurred while saving the model to '{file_name}.json'.")
            raise
        
        except Exception as e:
            # Captures any other unexpected error
            print(f"An unexpected error occurred: {str(e)}")
            raise Exception(f"Failed to save the model to '{file_name}.json'.") from e
        

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
        try:
            # Ensure the file has the .json extension
            if not file_name.endswith('.json'):
                file_name += '.json'
            
            # Open and load the JSON file
            with open(file_name, 'r') as file:
                data = json.load(file)
            
            print(f"Successfully loaded JSON file: {file_name}")
            return data
        
        except FileNotFoundError:
            print(f"Error: The file '{file_name}' was not found.")
            raise
        
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON file '{file_name}'. {str(e)}")
            raise
        
        except IOError as e:
            print(f"Error: An I/O error occurred while reading '{file_name}'. {str(e)}")
            raise