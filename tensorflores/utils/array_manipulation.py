import numpy as np

class ArrayManipulation:
    """
    A utility class for common operations like converting arrays to lists, 
    reshaping arrays based on shapes, and retrieving shapes of arrays.
    """

    @staticmethod
    def creat_list_from_array(arrayToTransform):
        """
        Converts a nested array structure into a flattened list while maintaining order.

        Args:
            arrayToTransform (list or np.ndarray): The input array to be transformed.

        Returns:
            np.ndarray: A 2D NumPy array with the flattened elements.

        Raises:
            ValueError: If the input array is empty or None.
            TypeError: If the input is not a list or NumPy array.
            RuntimeError: For unexpected errors during the transformation.
        """
        try:
            if arrayToTransform is None or len(arrayToTransform) == 0:
                raise ValueError("The input array cannot be None or empty.")
            if not isinstance(arrayToTransform, (list, np.ndarray)):
                raise TypeError("The input must be a list or a NumPy array.")
            
            flattened_list = []
            for input_element in arrayToTransform:
                try:
                    concatenated = np.concatenate(input_element).tolist()
                    flattened_list.append(concatenated)
                except Exception as e:
                    raise RuntimeError(f"Error concatenating input element: {str(e)}")
            
            # Flatten and reshape the list into a 2D array
            return np.array([item for sublist in flattened_list for item in sublist]).reshape(-1, 1)
        except Exception as e:
            raise RuntimeError(f"An error occurred in 'creat_list_from_array': {str(e)}")

    @staticmethod
    def creat_array_from_list(arrayToTransform, shapes):
        """
        Reshapes a flattened array into a list of arrays based on specified shapes.

        Args:
            arrayToTransform (np.ndarray): The input flattened array.
            shapes (list): A list of shapes to reshape the array into.

        Returns:
            list: A list of reshaped arrays.

        Raises:
            ValueError: If the input array or shapes list is empty or None.
            TypeError: If the inputs are not valid NumPy arrays or lists.
            RuntimeError: For unexpected errors during reshaping.
        """
        try:
            if arrayToTransform is None or len(arrayToTransform) == 0:
                raise ValueError("The input array cannot be None or empty.")
            if shapes is None or len(shapes) == 0:
                raise ValueError("The shapes list cannot be None or empty.")
            if not isinstance(arrayToTransform, np.ndarray):
                raise TypeError("The input array must be a NumPy array.")
            if not isinstance(shapes, list):
                raise TypeError("The shapes must be a list of tuples or lists.")

            result = []
            start = 0
            for shape in shapes:
                try:
                    size = np.prod(shape)
                    end = start + size
                    reshaped_array = arrayToTransform[start:end].reshape(shape)
                    result.append(reshaped_array)
                    start = end
                except Exception as e:
                    raise RuntimeError(f"Error reshaping array to shape {shape}: {str(e)}")

            return result
        except Exception as e:
            raise RuntimeError(f"An error occurred in 'convert_array_to_list': {str(e)}")

    @staticmethod
    def get_shape_from_array(input_array):
        """
        Retrieves the shapes of all arrays within a given list or array.

        Args:
            input_array (list or np.ndarray): A list or array containing other arrays.

        Returns:
            list: A list of shapes for each array in the input.

        Raises:
            ValueError: If the input array is empty or None.
            TypeError: If the input is not a list or NumPy array.
            RuntimeError: For unexpected errors during shape retrieval.
        """
        try:
            if input_array is None or len(input_array) == 0:
                raise ValueError("The input array cannot be None or empty.")
            if not isinstance(input_array, (list, np.ndarray)):
                raise TypeError("The input must be a list or a NumPy array.")

            result = []
            for array in input_array:
                try:
                    result.append(array.shape)
                except Exception as e:
                    raise RuntimeError(f"Error retrieving shape of array: {str(e)}")
            
            return result
        except Exception as e:
            raise RuntimeError(f"An error occurred in 'get_shape_from_array': {str(e)}")


    def convert_float32_to_int8(self, float_array, min_value, max_value):
        """
        Converts a NumPy array of float32 values to uint8 by normalizing 
        based on specified minimum and maximum values and scaling to the range [0, 255].

        Parameters:
        float_array: ndarray
            A NumPy array of float32 values to be quantized.
        min_value: float
            The minimum value for normalization.
        max_value: float
            The maximum value for normalization.

        Returns:
        ndarray:
            A NumPy array of uint8 values after quantization.

        Raises:
        ValueError:
            If min_value or max_value is not defined, or if they are equal.
        TypeError:
            If the input is not a valid NumPy array.
        """
        try:
            # Ensure min_value and max_value are defined
            if min_value is None or max_value is None:
                raise ValueError("Minimum and maximum values are not defined.")

            # Ensure max_value and min_value are different to avoid division by zero
            if max_value == min_value:
                raise ValueError("max_value and min_value are equal, making normalization impossible.")

            # Ensure the input is a NumPy array with dtype float32
            float_array = np.array(float_array, dtype=np.float32)

            # Normalize and quantize to uint8
            scaled_array = ((float_array - min_value) / (max_value - min_value)) * 255
            uint8_array = np.clip(scaled_array, 0, 255).astype(np.uint8)  # Ensure values stay within uint8 range

            return uint8_array

        except TypeError as e:
            print(f"Type error: {e}")  # Handle type-related errors
        except ValueError as e:
            print(f"Value error: {e}")  # Handle value-related errors
        except Exception as e:
            print(f"Unexpected error: {e}")  # Handle any other unexpected errors