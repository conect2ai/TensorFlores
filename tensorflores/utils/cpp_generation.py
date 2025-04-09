class CppGeneration:
    """
    The `CppGeneration` class is designed to facilitate the generation of C++ code for machine learning models, 
    specifically Multilayer Perceptrons (MLPs), from JSON-formatted model data. This class supports both quantized 
    and non-quantized models and provides methods to translate model parameters and architecture into functional 
    C++ code.

    Key Features:
    --------------
    - **Quantization Support**: Handles models trained with Post-Training Quantization (PTQ) or Quantization-Aware 
      Training (QAT), using quantized representations of weights and biases (e.g., centers and indices).
    - **Activation Functions**: Dynamically generates C++ code for a variety of activation functions used in the 
      neural network, including ReLU, Sigmoid, Tanh, and others.
    - **Matrix Multiplication**: Implements forward propagation through the MLP by generating C++ code for 
      efficient matrix multiplications and activation function applications.
    - **Error Handling**: Incorporates robust exception handling to manage potential issues during the generation 
      process.

    Methods:
    ---------
    - `__init__`: Initializes the class. Currently, no parameters are required for initialization.
    - `__generate_cpp_from_json_QAT(json_data)`: Converts a quantized neural network (QAT) described in JSON format 
      into C++ code. This includes special handling for quantized weights and biases.
    - `__generate_cpp_from_json_PQT(json_data, quantization_type)`: Converts a neural network using Post-Training 
      Quantization (PQT) into C++ code. Supports various quantization types for deployment in resource-constrained 
      environments.

    Usage:
    -------
    1. Provide JSON data representing the neural network model, including layers, weights, biases, and activation 
       functions.
    2. Call the appropriate method (`__generate_cpp_from_json_QAT` or `__generate_cpp_from_json_PQT`) to generate 
       C++ code tailored for the model type.
    3. The generated C++ code can be used for deployment on embedded systems, IoT devices, or other platforms 
       requiring compact and efficient neural network implementations.

    Raises:
    -------
    - `Exception`: For any unexpected issues during the C++ code generation process.

    Example:
    --------
    ```python
    cpp_gen = CppGeneration()
    json_data = {
        "model_quantized": True,
        "centers_bias": [0.1, 0.2, 0.3],
        "centers_weights": [0.5, 0.6, 0.7],
        "layers": [
            {
                "weights": [[1, 2], [3, 4]],
                "biases": [1, 1],
                "activation": "relu"
            }
        ]
    }
    cpp_code = cpp_gen.__generate_cpp_from_json_QAT(json_data)
    print(cpp_code)
    ```
    """


    def __init__(self):
        """
        Initializes the Clustering object. Currently, no specific initialization parameters are required.
        """
        pass


    def __generate_cpp_from_json_QAT(self, json_data):
        """
        Generates C++ code for a neural network (Multilayer Perceptron) based on the given JSON data.
        The function creates a class `MultilayerPerceptron` with a `predict` function that accepts an 
        input array and performs matrix multiplication for each layer.

        The C++ code includes the following:
        - Arrays for weights and biases for each layer.
        - Functions for various activation functions used in the layers.
        - Matrix multiplication code for performing forward propagation through the layers.
        
        The function handles two cases:
        - If the model is trained with quantization, it uses quantized weights and biases (stored as centers).
        - If the model is not trained with quantization, it uses the original weights and biases.

        Args:
            json_data (dict): The JSON data containing the model parameters (weights, biases, and activation functions).

        Returns:
            str: A string containing the generated C++ code.
            
        Raises:
            Exception: If any error occurs during the generation of the C++ code.
        """

        try:
            cpp_code = "namespace Conect2AI {\n"
            cpp_code += "namespace TensorFlores {\n"
            cpp_code += "class MultilayerPerceptron {\n"
            cpp_code += "public: \n\n"

            cpp_code += "float predict(float *x) { \n"
            cpp_code += "float y_pred = 0;\n"


            if (json_data['model_quantized']  == True) or (json_data['model_quantized']  == 'evolving'):
                # Generate centers array
                cpp_code += f"static const float center_bias[{len(json_data['centers_bias'])}] = "
                cpp_code += "{" + ", ".join(map(str, json_data["centers_bias"])) + "};\n\n"

                cpp_code += f"static const float centers_weights[{len(json_data['centers_weights'])}] = "
                cpp_code += "{" + ", ".join(map(str, json_data["centers_weights"])) + "};\n\n"



                # Generate positions of weights and bias arrays
                for i, layer in enumerate(json_data["layers"]):
                    if "weights" in layer:
                        cpp_code += "static const uint8_t w{0}[{1}][{2}] = ".format(i+1, len(layer["weights"]), len(layer["weights"][0]))
                        cpp_code += "{\n"
                        for row in layer["weights"]:
                            cpp_code += "    {" + ", ".join(map(str, row)) + "},\n"
                        cpp_code = cpp_code.rstrip(",\n") + "\n};\n\n"
                    if "biases" in layer:
                        cpp_code += "static const uint8_t b{0}[{1}] = ".format(i+1, len(layer["biases"]))
                        cpp_code += "{" + ", ".join(map(str, layer["biases"])) + "};\n\n"

                # Matrix multiplication
                for i, layer in enumerate(json_data["layers"]):

                    if i == 0:
                        cpp_code += f"    // Input Layer \n"
                        cpp_code += f"    float z{i + 1}[{len(layer['biases'])}];\n"
                        cpp_code += f"    for (int i = 0; i < {len(layer['biases'])}; i++)\n"
                        cpp_code += "    {\n"
                        cpp_code += f"        z{i + 1}[i] = center_bias[b{i + 1}[i]];\n"
                        cpp_code += f"        for (int j = 0; j < {len(layer['weights'])}; j++)\n"
                        cpp_code += "        {\n"
                        cpp_code += f"            z{i + 1}[i] += x[j] * centers_weights[w{i + 1}[j][i]];\n"
                        cpp_code += "        }\n"
                        cpp_code += f"        z{i + 1}[i] = {layer['activation']}(z{i + 1}[i]);"
                        cpp_code += "    }\n\n"
                    if i > 0 and i < len(json_data["layers"])-1:
                        cpp_code += f"    // Hidden Layers \n"
                        cpp_code += f"    float z{i + 1}[{len(layer['biases'])}];\n"
                        cpp_code += f"    for (int i = 0; i < {len(layer['biases'])}; i++)\n"
                        cpp_code += "    {\n"
                        cpp_code += f"        z{i + 1}[i] = center_bias[b{i + 1}[i]];\n"
                        cpp_code += f"        for (int j = 0; j < {len(layer['weights'])}; j++)\n"
                        cpp_code += "        {\n"
                        cpp_code += f"            z{i + 1}[i] += z{i}[j] * centers_weights[w{i + 1}[j][i]];\n"
                        cpp_code += "        }\n"
                        cpp_code += f"        z{i + 1}[i] = {layer['activation']}(z{i + 1}[i]);"
                        cpp_code += "    }\n\n"
                    if i == len(json_data["layers"])-1:
                        cpp_code += f"    // Output Layer\n"
                        cpp_code += f"    float z{i + 1} = center_bias[b{i + 1}[0]];\n"
                        cpp_code += f"    for (int i = 0; i < {len(layer['weights'])}; i++)\n"
                        cpp_code += "     {\n"
                        cpp_code += f"       z{i + 1} += z{i}[i] * centers_weights[w{i + 1}[i][0]];\n"
                        cpp_code += f"       z{i + 1} = {layer['activation']}(z{i + 1});"
                        cpp_code += "     }\n\n"  
                        
                cpp_code += "y_pred = z{0};\n".format(len(json_data["layers"]))
                cpp_code += "return y_pred;"
                cpp_code += "}\n"
    
            else:
                for i, layer in enumerate(json_data["layers"]):
                    if "weights" in layer:
                        cpp_code += "static const float w{0}[{1}][{2}] = ".format(i+1, len(layer["weights"]), len(layer["weights"][0]))
                        cpp_code += "{\n"
                        for row in layer["weights"]:
                            cpp_code += "    {" + ", ".join(map(str, row)) + "},\n"
                        cpp_code = cpp_code.rstrip(",\n") + "\n};\n\n"
                    if "biases" in layer:
                        cpp_code += "static const float b{0}[{1}] = ".format(i+1, len(layer["biases"]))
                        cpp_code += "{" + ", ".join(map(str, layer["biases"])) + "};\n\n"

                # Matrix multiplication
                for i, layer in enumerate(json_data["layers"]):

                    if i == 0:
                        cpp_code += f"    // Input Layer \n"
                        cpp_code += f"    float z{i + 1}[{len(layer['biases'])}];\n"
                        cpp_code += f"    for (int i = 0; i < {len(layer['biases'])}; i++)\n"
                        cpp_code += "    {\n"
                        cpp_code += f"        z{i + 1}[i] = b{i + 1}[i];\n"
                        cpp_code += f"        for (int j = 0; j < {len(layer['weights'])}; j++)\n"
                        cpp_code += "        {\n"
                        cpp_code += f"            z{i + 1}[i] += x[j] * w{i + 1}[j][i];\n"
                        cpp_code += "        }\n"
                        cpp_code += f"        z{i + 1}[i] = {layer['activation']}(z{i + 1}[i]);"
                        cpp_code += "    }\n\n"
                    if i > 0 and i < len(json_data["layers"])-1:
                        cpp_code += f"    // Hidden Layers \n"
                        cpp_code += f"    float z{i + 1}[{len(layer['biases'])}];\n"
                        cpp_code += f"    for (int i = 0; i < {len(layer['biases'])}; i++)\n"
                        cpp_code += "    {\n"
                        cpp_code += f"        z{i + 1}[i] = b{i + 1}[i];\n"
                        cpp_code += f"        for (int j = 0; j < {len(layer['weights'])}; j++)\n"
                        cpp_code += "        {\n"
                        cpp_code += f"            z{i + 1}[i] += z{i}[j] * w{i + 1}[j][i];\n"
                        cpp_code += "        }\n"
                        cpp_code += f"        z{i + 1}[i] = {layer['activation']}(z{i + 1}[i]);"
                        cpp_code += "    }\n\n"
                    if i == len(json_data["layers"])-1:
                        cpp_code += f"    // Output Layer\n"
                        cpp_code += f"    float z{i + 1} = b{i + 1}[0];\n"
                        cpp_code += f"    for (int i = 0; i < {len(layer['weights'])}; i++)\n"
                        cpp_code += "     {\n"
                        cpp_code += f"       z{i + 1} += z{i}[i] * w{i + 1}[i][0];\n"
                        cpp_code += f"       z{i + 1} = {layer['activation']}(z{i + 1});"
                        cpp_code += "     }\n\n"  
                        
                cpp_code += "y_pred = z{0};\n".format(len(json_data["layers"]))
                cpp_code += "return y_pred;"
                cpp_code += "}\n"


            cpp_code += "protected:\n"
            # Get all activation functions
            list_activation_function = []
            for layer in json_data["layers"]:
                list_activation_function.append(layer['activation'])
                

            # Remove dup and get only activation functions used
            for activation in set(list_activation_function):
            # Fuction activation equations
                if activation == 'relu':
                    cpp_code += "float relu(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    return x > 0 ? x : 0;\n"
                    cpp_code += "};\n\n"

                if activation == 'linear':
                    cpp_code += "float linear(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    return x;\n"
                    cpp_code += "};\n\n"

                if activation == 'sigmoid':
                    cpp_code += "float sigmoid(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    return 1.0 / (1.0 + exp(-x));\n"
                    cpp_code += "};\n\n"

                if activation == 'tanh':
                    cpp_code += "float tanh(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));\n"
                    cpp_code += "};\n\n"

                if activation == 'prelu':
                    cpp_code += "float prelu(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    float alpha = 0.01;\n"
                    cpp_code += "    return max(alpha * x, x);\n"
                    cpp_code += "};\n\n"

                if activation == 'leaky_relu':
                    cpp_code += "float leaky_relu(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    float alpha = 0.01;\n"
                    cpp_code += "    return (x > 0) ? x : alpha * x;\n"
                    cpp_code += "};\n\n"

                if activation == 'elu':
                    cpp_code += "float elu(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    float alpha = 0.01;\n"
                    cpp_code += "    return (x > 0) ? x : alpha * (exp(x) - 1);\n"
                    cpp_code += "};\n\n"

                if activation == 'selu':
                    cpp_code += "float selu(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    float scale = 1.0507;\n"
                    cpp_code += "    float alpha = 1.6732;\n"
                    cpp_code += "    return (x > 0) ? (scale * x) : (scale * alpha * (exp(x) - 1));\n"
                    cpp_code += "};\n\n"

                if activation == 'swish':
                    cpp_code += "float swish(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    return x / (1 + exp(-x));\n"
                    cpp_code += "};\n\n"

                if activation == 'gelu':
                    cpp_code += "float gelu(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    return 0.5 * x * (1 + tanh(0.797885 * (x + 0.044715 * pow(x, 3))));\n"
                    cpp_code += "};\n\n"

                if activation == 'softplus(':
                    cpp_code += "float softplus(float x)\n"
                    cpp_code += "{\n"
                    cpp_code += "    return log(1 + exp(x));\n"
                    cpp_code += "};\n\n"


            cpp_code += "};\n"
            cpp_code += "}\n"
            cpp_code += "}\n"
            
            return cpp_code
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}") from e
        

    def __generate_cpp_from_json_PQT(self, json_data, quantization_type:str = 'int8', quantization_scheme:str = 'total'):
        """
        Generates C++ code from a TensorFlow or Tensorflores model in JSON format.
        
        Parameters:
        json_data (dict): The model's information in JSON format, including weights, biases, and activations.
        quantization_type (str): Type of quantization ('float32' or 'int8').
        quantization_scheme (str): The quantization scheme ('total', 'weights', or 'biases').
        
        Returns:
        str: C++ code representing the model.
        
        Exceptions:
        - KeyError: If a key in the JSON data is missing.
        - ValueError: If the data has invalid values.
        - Exception: For any other unexpected errors.
        """   
        try:
            
            cpp_code = (
                "namespace Conect2AI {\n"
                "namespace TensorFlores {\n"
                "class MultilayerPerceptron {\n"
                "public: \n\n"
                "float predict(float *x) { \n"
                "float y_pred = 0;\n"
            )

            # Generate weights and bias arrays
            for i, layer in enumerate(json_data["layers"]):
                if "weights" in layer:
                    weight_type = "uint8_t" if quantization_type == 'int8' and quantization_scheme in ['total', 'weights'] else "float"
                    cpp_code += f"static const {weight_type} w{i+1}[{len(layer['weights'])}][{len(layer['weights'][0])}] = {{\n"
                    for row in layer["weights"]:
                        cpp_code += "    {" + ", ".join(map(str, row)) + "},\n"
                    cpp_code = cpp_code.rstrip(",\n") + "\n};\n\n"
                if "biases" in layer:
                    bias_type = "uint8_t" if quantization_type == 'int8' and quantization_scheme in ['total', 'biases'] else "float"
                    cpp_code += f"static const {bias_type} b{i+1}[{len(layer['biases'])}] = {{{', '.join(map(str, layer['biases']))}}};\n\n"

            # Matrix multiplication
            for i, layer in enumerate(json_data["layers"]):
                if i == 0:
                    cpp_code += f"    // Input Layer \n"
                    cpp_code += f"    float z{i + 1}[{len(layer['biases'])}];\n"
                    cpp_code += f"    for (int i = 0; i < {len(layer['biases'])}; i++)\n    {{\n"
                    bias_expr = f"dequantized(b{i + 1}[i])" if quantization_type == 'int8' and quantization_scheme in ['total', 'biases'] else f"b{i + 1}[i]"
                    cpp_code += f"        z{i + 1}[i] = {bias_expr};\n"
                    cpp_code += f"        for (int j = 0; j < {len(layer['weights'])}; j++)\n        {{\n"
                    weight_expr = f"x[j] * dequantized(w{i + 1}[j][i])" if quantization_type == 'int8' and quantization_scheme in ['total', 'weights'] else f"x[j] * w{i + 1}[j][i]"
                    cpp_code += f"            z{i + 1}[i] += {weight_expr};\n"
                    cpp_code += "        }\n"
                    cpp_code += f"        z{i + 1}[i] = {layer['activation']}(z{i + 1}[i]);\n"
                    cpp_code += "    }\n\n"

                elif i < len(json_data["layers"]) - 1:
                    cpp_code += f"    // Hidden Layer {i + 1}\n"
                    cpp_code += f"    float z{i + 1}[{len(layer['biases'])}];\n"
                    cpp_code += f"    for (int i = 0; i < {len(layer['biases'])}; i++)\n    {{\n"
                    bias_expr = f"dequantized(b{i + 1}[i])" if quantization_type == 'int8' and quantization_scheme in ['total', 'biases'] else f"b{i + 1}[i]"
                    cpp_code += f"        z{i + 1}[i] = {bias_expr};\n"
                    cpp_code += f"        for (int j = 0; j < {len(layer['weights'])}; j++)\n        {{\n"
                    weight_expr = f"z{i}[j] * dequantized(w{i + 1}[j][i])" if quantization_type == 'int8' and quantization_scheme in ['total', 'weights'] else f"z{i}[j] * w{i + 1}[j][i]"
                    cpp_code += f"            z{i + 1}[i] += {weight_expr};\n"
                    cpp_code += "        }\n"
                    cpp_code += f"        z{i + 1}[i] = {layer['activation']}(z{i + 1}[i]);\n"
                    cpp_code += "    }\n\n"

                else:
                    cpp_code += f"    // Output Layer\n"
                    bias_expr = f"dequantized(b{i + 1}[0])" if quantization_type == 'int8' and quantization_scheme in ['total', 'biases'] else f"b{i + 1}[0]"
                    cpp_code += f"    float z{i + 1} = {bias_expr};\n"
                    cpp_code += f"    for (int i = 0; i < {len(layer['weights'])}; i++)\n    {{\n"
                    weight_expr = f"z{i}[i] * dequantized(w{i + 1}[i][0])" if quantization_type == 'int8' and quantization_scheme in ['total', 'weights'] else f"z{i}[i] * w{i + 1}[i][0]"
                    cpp_code += f"        z{i + 1} += {weight_expr};\n"
                    cpp_code += f"        z{i + 1} = {layer['activation']}(z{i + 1});\n"
                    cpp_code += "    }\n\n"

            cpp_code += f"y_pred = z{len(json_data['layers'])};\n"
            cpp_code += "return y_pred;\n"
            cpp_code += "}\nprotected:\n"

            # Add dequantization function if needed
            if quantization_type == 'int8' and quantization_scheme in ['total', 'weights', 'biases']:
                cpp_code += (
                    "float dequantized(uint8_t x)\n"
                    "{\n"
                    f"    return (((x) / 255.0) * ({json_data['max_value']} - {json_data['min_value']}) + {json_data['min_value']});\n"
                    "};\n\n"
                )

            # Generate activation functions
            activations = set(layer['activation'] for layer in json_data["layers"])
            activation_functions = {
                'relu': "float relu(float x)\n{\n    return x > 0 ? x : 0;\n};\n\n",
                'linear': "float linear(float x)\n{\n    return x;\n};\n\n",
                'sigmoid': "float sigmoid(float x)\n{\n    return 1.0 / (1.0 + exp(-x));\n};\n\n",
                'tanh': "float tanh(float x)\n{\n    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));\n};\n\n",
                'prelu': "float prelu(float x)\n{\n    float alpha = 0.01;\n    return max(alpha * x, x);\n};\n\n",
                'leaky_relu': "float leaky_relu(float x)\n{\n    float alpha = 0.01;\n    return (x > 0) ? x : alpha * x;\n};\n\n",
                'elu': "float elu(float x)\n{\n    float alpha = 0.01;\n    return (x > 0) ? x : alpha * (exp(x) - 1);\n};\n\n",
                'selu': "float selu(float x)\n{\n    float scale = 1.0507;\n    float alpha = 1.6732;\n    return (x > 0) ? (scale * x) : (scale * alpha * (exp(x) - 1));\n};\n\n",
                'swish': "float swish(float x)\n{\n    return x / (1 + exp(-x));\n};\n\n",
                'gelu': "float gelu(float x)\n{\n    return 0.5 * x * (1 + tanh(0.797885 * (x + 0.044715 * pow(x, 3))));\n};\n\n",
                'softplus': "float softplus(float x)\n{\n    return log(1 + exp(x));\n};\n\n"
            }

            for activation in activations:
                if activation in activation_functions:
                    cpp_code += activation_functions[activation]

            cpp_code += "};\n"  # End of class
            cpp_code += "}\n}\n"  # End of namespaces
            
            return cpp_code

        except KeyError as e:
            print(f"Key error: {e}")
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")





    def __generate_cpp_from_json_not_quant(self, json_data):
        """
        Generates C++ code from a TensorFlow or Tensorflores model in JSON format.
        
        Parameters:
        json_data (dict): The model's information in JSON format, including weights, biases, and activations.
        quantization_type (str): Type of quantization ('float32' or 'int8').
        quantization_scheme (str): The quantization scheme ('total', 'weights', or 'biases').
        
        Returns:
        str: C++ code representing the model.
        
        Exceptions:
        - KeyError: If a key in the JSON data is missing.
        - ValueError: If the data has invalid values.
        - Exception: For any other unexpected errors.
        """   
        try:
            
            cpp_code = (
                "namespace Conect2AI {\n"
                "namespace TensorFlores {\n"
                "class MultilayerPerceptron {\n"
                "public: \n\n"
                "float predict(float *x) { \n"
                "float y_pred = 0;\n"
            )

            # Generate weights and bias arrays
            for i, layer in enumerate(json_data["layers"]):
                if "weights" in layer:
                    cpp_code += f"static const float w{i+1}[{len(layer['weights'])}][{len(layer['weights'][0])}] = {{\n"
                    for row in layer["weights"]:
                        cpp_code += "    {" + ", ".join(map(str, row)) + "},\n"
                    cpp_code = cpp_code.rstrip(",\n") + "\n};\n\n"
                if "biases" in layer:
                    cpp_code += f"static const float b{i+1}[{len(layer['biases'])}] = {{{', '.join(map(str, layer['biases']))}}};\n\n"

            # Matrix multiplication
            for i, layer in enumerate(json_data["layers"]):
                if i == 0:
                    cpp_code += f"    // Input Layer \n"
                    cpp_code += f"    float z{i + 1}[{len(layer['biases'])}];\n"
                    cpp_code += f"    for (int i = 0; i < {len(layer['biases'])}; i++)\n    {{\n"
                    bias_expr = f"b{i + 1}[i]"
                    cpp_code += f"        z{i + 1}[i] = {bias_expr};\n"
                    cpp_code += f"        for (int j = 0; j < {len(layer['weights'])}; j++)\n        {{\n"
                    weight_expr =  f"x[j] * w{i + 1}[j][i]"
                    cpp_code += f"            z{i + 1}[i] += {weight_expr};\n"
                    cpp_code += "        }\n"
                    cpp_code += f"        z{i + 1}[i] = {layer['activation']}(z{i + 1}[i]);\n"
                    cpp_code += "    }\n\n"

                elif i < len(json_data["layers"]) - 1:
                    cpp_code += f"    // Hidden Layer {i + 1}\n"
                    cpp_code += f"    float z{i + 1}[{len(layer['biases'])}];\n"
                    cpp_code += f"    for (int i = 0; i < {len(layer['biases'])}; i++)\n    {{\n"
                    bias_expr = f"b{i + 1}[i]"
                    cpp_code += f"        z{i + 1}[i] = {bias_expr};\n"
                    cpp_code += f"        for (int j = 0; j < {len(layer['weights'])}; j++)\n        {{\n"
                    weight_expr =  f"z{i}[j] * w{i + 1}[j][i]"
                    cpp_code += f"            z{i + 1}[i] += {weight_expr};\n"
                    cpp_code += "        }\n"
                    cpp_code += f"        z{i + 1}[i] = {layer['activation']}(z{i + 1}[i]);\n"
                    cpp_code += "    }\n\n"

                else:
                    cpp_code += f"    // Output Layer\n"
                    bias_expr = f"b{i + 1}[0]"
                    cpp_code += f"    float z{i + 1} = {bias_expr};\n"
                    cpp_code += f"    for (int i = 0; i < {len(layer['weights'])}; i++)\n    {{\n"
                    weight_expr = f"z{i}[i] * w{i + 1}[i][0]"
                    cpp_code += f"        z{i + 1} += {weight_expr};\n"
                    cpp_code += f"        z{i + 1} = {layer['activation']}(z{i + 1});\n"
                    cpp_code += "    }\n\n"

            cpp_code += f"y_pred = z{len(json_data['layers'])};\n"
            cpp_code += "return y_pred;\n"
            cpp_code += "}\nprotected:\n"

   
            # Generate activation functions
            activations = set(layer['activation'] for layer in json_data["layers"])
            activation_functions = {
                'relu': "float relu(float x)\n{\n    return x > 0 ? x : 0;\n};\n\n",
                'linear': "float linear(float x)\n{\n    return x;\n};\n\n",
                'sigmoid': "float sigmoid(float x)\n{\n    return 1.0 / (1.0 + exp(-x));\n};\n\n",
                'tanh': "float tanh(float x)\n{\n    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));\n};\n\n",
                'prelu': "float prelu(float x)\n{\n    float alpha = 0.01;\n    return max(alpha * x, x);\n};\n\n",
                'leaky_relu': "float leaky_relu(float x)\n{\n    float alpha = 0.01;\n    return (x > 0) ? x : alpha * x;\n};\n\n",
                'elu': "float elu(float x)\n{\n    float alpha = 0.01;\n    return (x > 0) ? x : alpha * (exp(x) - 1);\n};\n\n",
                'selu': "float selu(float x)\n{\n    float scale = 1.0507;\n    float alpha = 1.6732;\n    return (x > 0) ? (scale * x) : (scale * alpha * (exp(x) - 1));\n};\n\n",
                'swish': "float swish(float x)\n{\n    return x / (1 + exp(-x));\n};\n\n",
                'gelu': "float gelu(float x)\n{\n    return 0.5 * x * (1 + tanh(0.797885 * (x + 0.044715 * pow(x, 3))));\n};\n\n",
                'softplus': "float softplus(float x)\n{\n    return log(1 + exp(x));\n};\n\n"
            }

            for activation in activations:
                if activation in activation_functions:
                    cpp_code += activation_functions[activation]

            cpp_code += "};\n"  # End of class
            cpp_code += "}\n}\n"  # End of namespaces
            
            return cpp_code

        except KeyError as e:
            print(f"Key error: {e}")
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")



    def  generate_cpp_from_json(self, json_data, file_name:str):
        """
        Generates C++ code from a TensorFlow or Multilayer perceptron model in JSON format.
        
        Parameters:
        json_data (dict): The model's information in JSON format, including weights, biases, and activations.
        quantization_type (str): Type of quantization ('float32' or 'int8').
        quantization_scheme (str): The quantization scheme ('total', 'weights', or 'biases').
        
        Returns:
        str: C++ code representing the model.
        
        Exceptions:
        - KeyError: If a key in the JSON data is missing.
        - ValueError: If the data has invalid values.
        - Exception: For any other unexpected errors.
        """   
        if json_data['model_quantized'] == "evolving":

            cpp_model = CppGeneration().__generate_cpp_from_json_QAT(json_data = json_data)
        elif json_data['model_quantized'] == "int8":
            cpp_model = CppGeneration().__generate_cpp_from_json_PQT(json_data = json_data)
        else:
            cpp_model = CppGeneration().__generate_cpp_from_json_not_quant(json_data)
        with open(f'{file_name}.h', 'w') as file:
            file.write(cpp_model)
            
        # Confirm that the model has been saved successfully
        print('Model C++ saved!')

        return cpp_model 
    



    def save_model_as_cpp(self, file_name: str, json_model):
        """
        Saves the model to a .h file with the specified file name.
        
        Parameters:
        file_name (str): The name of the file where the model will be saved.
        
        Exceptions:
        - If an error occurs while extracting the model parameters or generating the C++ code, an exception will be raised with the error message.
        - If the file cannot be opened or written to, an exception will be raised.
        """
        try:           
            # Generate the C++ code for the model
            cpp_model = CppGeneration().__generate_cpp_from_json_QAT(json_data = json_model)
            
            # Save the generated code to a .h file
            with open(f'{file_name}.h', 'w') as file:
                file.write(cpp_model)
            
            # Confirm that the model has been saved successfully
            print('Model C++ saved!')
        
        except FileNotFoundError:
            # Error if the file cannot be found or opened for writing
            print(f"Error: The file '{file_name}.h' could not be found or opened for writing.")
        
        except IOError:
            # Generic I/O error
            print(f"Error: An I/O error occurred while saving the model '{file_name}.h'.")
        
        except Exception as e:
            # Captures any other unexpected error
            print(f"An unexpected error occurred: {str(e)}")
            raise Exception(f"Failed to save model to '{file_name}.h'.") from e