import numpy as np
from typing import Dict, Optional, List, Any
import copy

from tensorflores.models.multilayer_perceptron import MultilayerPerceptron
from tensorflores.utils.array_manipulation import ArrayManipulation
from tensorflores.utils.json_handle import JsonHandle


class Quantization:
    """
    A class for performing clustering operations using various algorithms such as AutoCloud, 
    MeanShift, Affinity Propagation, and DBSTREAM. Provides methods for clustering both weights 
    and biases with specified thresholds and parameters.
    """

    def __init__(self):
        """
        Initializes the Quantization object. Currently, no specific initialization parameters are required.
        """
        pass

    def post_training_quantization(
        self, 
        json_data: Dict[str, Any], 
        quantization_type: str, 
        distance_metric: Optional[str] = None, 
        bias_clustering_method: Optional[str] = None, 
        weight_clustering_method: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Performs post-training quantization based on specified methods.

        Args:
            json_data (Dict[str, Any]): JSON data containing model details.
            quantization_type (str): Type of quantization ('int8' or 'evolving').
            distance_metric (Optional[str]): Metric used for clustering distance.
            bias_clustering_method (Optional[str]): Method for bias clustering.
            weight_clustering_method (Optional[str]): Method for weight clustering.

        Returns:
            Optional[Dict[str, Any]]: Modified JSON data with applied quantization, or None if an error occurs.

        Raises:
            ValueError: If an error occurs during quantization.
        """
        try:
            # Faz uma cópia profunda do json_data para evitar alterar o objeto original
            json_data_copy = copy.deepcopy(json_data)

            weights, weights_shape, biases, biases_shape = JsonHandle().paramters_from_json(json_data_copy)
            weight_list = ArrayManipulation().creat_list_from_array(weights)
            bias_list = ArrayManipulation().creat_list_from_array(biases)

            if quantization_type == 'int8':
                return self._apply_int8_quantization(json_data_copy, weight_list, bias_list, weights_shape, biases_shape)
            elif quantization_type == "evolving":
                return self._apply_evolving_quantization(
                    json_data_copy, weight_list, bias_list, weights_shape, biases_shape, 
                    distance_metric, bias_clustering_method, weight_clustering_method
                )
            else:
                print(f"{quantization_type} is an invalid method for post-training quantization.")
                return None

        except Exception as e:
            raise ValueError(f"Error in post-training quantization: {e}.")

    def _apply_int8_quantization(
        self, 
        json_data: Dict[str, Any], 
        weight_list: List[float], 
        bias_list: List[float], 
        weights_shape: tuple, 
        biases_shape: tuple
    ) -> Dict[str, Any]:
        """
        Applies int8 quantization to the model weights and biases.

        Args:
            json_data (Dict[str, Any]): JSON data containing model details.
            weight_list (List[float]): List of weight values.
            bias_list (List[float]): List of bias values.
            weights_shape (tuple): Shape of the weights array.
            biases_shape (tuple): Shape of the biases array.

        Returns:
            Dict[str, Any]: Modified JSON data with int8 quantization applied.
        """
        max_value = np.max([np.max(weight_list), np.max(bias_list)])
        min_value = np.min([np.min(weight_list), np.min(bias_list)])

        weights_int8_quant = ArrayManipulation().creat_array_from_list(
            ArrayManipulation().convert_float32_to_int8(weight_list, min_value, max_value), weights_shape
        )
        biases_int8_quant = ArrayManipulation().creat_array_from_list(
            ArrayManipulation().convert_float32_to_int8(bias_list, min_value, max_value), biases_shape
        )

        return JsonHandle().int8_scheme_change(json_data, weights_int8_quant, biases_int8_quant, min_value, max_value)

    def _apply_evolving_quantization(
        self, 
        json_data: Dict[str, Any], 
        weight_list: List[float], 
        bias_list: List[float], 
        weights_shape: tuple, 
        biases_shape: tuple, 
        distance_metric: Optional[str], 
        bias_clustering_method: Optional[str], 
        weight_clustering_method: Optional[str]
    ) -> Dict[str, Any]:
        """
        Applies evolving quantization to the model weights and biases.

        Args:
            json_data (Dict[str, Any]): JSON data containing model details.
            weight_list (List[float]): List of weight values.
            bias_list (List[float]): List of bias values.
            weights_shape (tuple): Shape of the weights array.
            biases_shape (tuple): Shape of the biases array.
            distance_metric (Optional[str]): Metric used for clustering distance.
            bias_clustering_method (Optional[str]): Method for bias clustering.
            weight_clustering_method (Optional[str]): Method for weight clustering.

        Returns:
            Dict[str, Any]: Modified JSON data with evolving quantization applied.
        """

        mlp = MultilayerPerceptron()

        for i in range(10):
            bias_center = mlp.applying_clusterings(
                clustering_method=bias_clustering_method, parameter_list=bias_list
            )
            weight_center = mlp.applying_clusterings(
                clustering_method=weight_clustering_method, parameter_list=weight_list
            )

        result_df_bias = mlp.find_closest_values(
            bias_list, bias_center, distance_metric=distance_metric
        )
        result_df_weight = mlp.find_closest_values(
            weight_list, weight_center, distance_metric=distance_metric
        )

        weights_index = ArrayManipulation().creat_array_from_list(
            result_df_weight['Index'].values, weights_shape
        )
        biases_index = ArrayManipulation().creat_array_from_list(
            result_df_bias['Index'].values, biases_shape
        )

        json_data['centers_bias'] = bias_center
        json_data['centers_weights'] = weight_center
        json_data['model_quantized'] = "evolving"

        for i in range(len(json_data['layers'])):
            json_data['layers'][i]['weights'] = weights_index[i].tolist()
            json_data['layers'][i]['biases'] = biases_index[i].tolist()

        return json_data