from tensorflores.utils.autocloud.auto_cloud_bias import AutoCloudBias 
from tensorflores.utils.autocloud.auto_cloud_weight import AutoCloudWeight
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from river import cluster


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
