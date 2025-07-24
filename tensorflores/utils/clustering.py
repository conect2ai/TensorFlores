from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from river import cluster
import numpy as np
from river import stream
import pandas as pd

from tensorflores.utils.autocloud.auto_cloud_bias import AutoCloudBias 
from tensorflores.utils.autocloud.auto_cloud_weight import AutoCloudWeight



class ClusteringMethods:
    """
    A class for performing clustering operations using various algorithms such as AutoCloud, 
    MeanShift, Affinity Propagation, and DBSTREAM. Provides methods for clustering both weights 
    and biases with specified thresholds and parameters.
    """


    def __init__(self):
        """
        Initializes the Clustering object. Currently, no specific initialization parameters are required.
        """
        pass

    def autocloud_weight(self, threshold_weights: float = 1.4148):
        """
        Applies the AutoCloud algorithm for clustering weights.

        Args:
            threshold_weights (float): The threshold value for clustering weights.

        Returns:
            tuple: A tuple containing the algorithm name ('AUTOCLOUD') and the AutoCloudWeight object.
        """
        try:
            autocloud_weight = AutoCloudWeight(m=threshold_weights)
            return 'AUTOCLOUD', autocloud_weight
        except Exception as e:
            raise RuntimeError(f"Error in autocloud_weight: {e}")
        

    def autocloud_biases(self, threshold_biases: float = 1.415):
        """
        Applies the AutoCloud algorithm for clustering biases.

        Args:
            threshold_biases (float): The threshold value for clustering biases.

        Returns:
            tuple: A tuple containing the algorithm name ('AUTOCLOUD') and the AutoCloudBias object.
        """
        try:
            autocloud_biases = AutoCloudBias(m=threshold_biases)
            return 'AUTOCLOUD', autocloud_biases
        except Exception as e:
            raise RuntimeError(f"Error in autocloud_biases: {e}")
        

    def meanshift_weight(self, bandwidth_weights: float = 0.005, max_iter: int = 300, bin_seeding: bool = True):
        """
        Applies the MeanShift algorithm for clustering weights.

        Args:
            bandwidth_weights (float): Bandwidth parameter for the algorithm.
            max_iter (int): Maximum number of iterations.
            bin_seeding (bool): Whether to seed initial bin locations.

        Returns:
            tuple: A tuple containing the algorithm name ('MEANSHIFT') and the MeanShift object.
        """
        try:
            meanshift_weight = MeanShift(bandwidth=bandwidth_weights, max_iter=max_iter, bin_seeding=bin_seeding)
            return 'MEANSHIFT', meanshift_weight
        except Exception as e:
            raise RuntimeError(f"Error in meanshift_weight: {e}")
        

    def meanshift_biases(self, bandwidth_biases: float = 0.005, max_iter: int = 300, bin_seeding: bool = True):
        """
        Applies the MeanShift algorithm for clustering biases.

        Args:
            bandwidth_biases (float): Bandwidth parameter for the algorithm.
            max_iter (int): Maximum number of iterations.
            bin_seeding (bool): Whether to seed initial bin locations.

        Returns:
            tuple: A tuple containing the algorithm name ('MEANSHIFT') and the MeanShift object.
        """
        try:
            meanshift_biases = MeanShift(bandwidth=bandwidth_biases, max_iter=max_iter, bin_seeding=bin_seeding)
            return 'MEANSHIFT', meanshift_biases
        except Exception as e:
            raise RuntimeError(f"Error in meanshift_biases: {e}")

    def affinity_propagation_weight(self, affinityprop_damping_weight: float = 0.7, random_state: int = 42, max_iter: int = 500, convergence_iter: int = 20):
        """
        Applies the Affinity Propagation algorithm for clustering weights.

        Args:
            affinityprop_damping_weight (float): Damping factor for the algorithm.
            random_state (int): Random state for reproducibility.
            max_iter (int): Maximum number of iterations.
            convergence_iter (int): Number of iterations with no change for convergence.

        Returns:
            tuple: A tuple containing the algorithm name ('AFFINITYPROP') and the AffinityPropagation object.
        """
        try:
            affinity_propagation_weight = AffinityPropagation(
                damping=affinityprop_damping_weight,
                random_state=random_state,
                max_iter=max_iter,
                convergence_iter=convergence_iter
            )
            return 'AFFINITYPROP', affinity_propagation_weight
        except Exception as e:
            raise RuntimeError(f"Error in affinity_propagation_weight: {e}")
        

    def affinity_propagation_biases(self, affinityprop_damping_bias: float = 0.65, random_state: int = 42, max_iter: int = 500, convergence_iter: int = 20):
        """
        Applies the Affinity Propagation algorithm for clustering biases.

        Args:
            affinityprop_damping_bias (float): Damping factor for the algorithm.
            random_state (int): Random state for reproducibility.
            max_iter (int): Maximum number of iterations.
            convergence_iter (int): Number of iterations with no change for convergence.

        Returns:
            tuple: A tuple containing the algorithm name ('AFFINITYPROP') and the AffinityPropagation object.
        """
        try:
            affinity_propagation_biases = AffinityPropagation(
                damping=affinityprop_damping_bias,
                random_state=random_state,
                max_iter=max_iter,
                convergence_iter=convergence_iter
            )
            return 'AFFINITYPROP', affinity_propagation_biases
        except Exception as e:
            raise RuntimeError(f"Error in affinity_propagation_biases: {e}")
        

    def dbstream_weight(self, clustering_threshold_weight: float = 0.1, fading_factor: float = 0.05, cleanup_interval: int = 4, intersection_factor: float = 0.5, minimum_weight: int = 1):
        """
        Applies the DBSTREAM algorithm for clustering weights.

        Args:
            clustering_threshold_weight (float): Threshold for clustering.
            fading_factor (float): Fading factor for the algorithm.
            cleanup_interval (int): Interval for cleanup operations.
            intersection_factor (float): Factor for handling intersections.
            minimum_weight (int): Minimum weight threshold.

        Returns:
            tuple: A tuple containing the algorithm name ('DBSTREAM') and the DBSTREAM object.
        """
        try:
            dbstream_weight = cluster.DBSTREAM(
                clustering_threshold=clustering_threshold_weight,
                fading_factor=fading_factor,
                cleanup_interval=cleanup_interval,
                intersection_factor=intersection_factor,
                minimum_weight=minimum_weight
            )
            return 'DBSTREAM', dbstream_weight
        except Exception as e:
            raise RuntimeError(f"Error in dbstream_weight: {e}")
        

    def dbstream_biases(self, clustering_threshold_bias: float = 0.8, fading_factor: float = 0.05, cleanup_interval: int = 4, intersection_factor: float = 0.5, minimum_weight: int = 1):
        """
        Applies the DBSTREAM algorithm for clustering biases.

        Args:
            clustering_threshold_bias (float): Threshold for clustering.
            fading_factor (float): Fading factor for the algorithm.
            cleanup_interval (int): Interval for cleanup operations.
            intersection_factor (float): Factor for handling intersections.
            minimum_weight (int): Minimum weight threshold.

        Returns:
            tuple: A tuple containing the algorithm name ('DBSTREAM') and the DBSTREAM object.
        """
        try:
            dbstream_biases = cluster.DBSTREAM(
                clustering_threshold=clustering_threshold_bias,
                fading_factor=fading_factor,
                cleanup_interval=cleanup_interval,
                intersection_factor=intersection_factor,
                minimum_weight=minimum_weight
            )
            return 'DBSTREAM', dbstream_biases
        except Exception as e:
            raise RuntimeError(f"Error in dbstream_biases: {e}")


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