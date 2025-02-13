&nbsp;
<p align="center">
  <img width="400" src="https://github.com/conect2ai/Conect2Py-Package/assets/56210040/60055d32-77f0-4381-bfc1-c9300eb30920" />
</p> 
&nbsp;
&nbsp;
<p align="center">
  <img width="800" src="https://drive.google.com/uc?export=view&id=1JlRnq5IG1ZMwfzu2-Wr9rKw5Hd-MidIn" />
</p> 
&nbsp;






# TensorFlores: An Enhanced Python-based TinyML Framework


The TensorFlores framework is a Python-based solution designed for optimizing machine learning deployment in resource-constrained environments. It introduces an evolving clustering-based quantization, enabling quantization-aware training (QAT) and post-training quantization (PTQ) while preserving model accuracy. TensorFlores seamlessly converts TensorFlow models into optimized formats and generates platform-agnostic C++ code for embedded systems. Its modular architecture minimizes memory usage and computational overhead, ensuring efficient real-time inference. By integrating clustering-based quantization and automated code generation, TensorFlores enhances the feasibility of TinyML applications, particularly in low-power and edge AI scenarios. This framework provides a robust and scalable solution for deploying machine learning models in embedded and IoT systems.

<p align="right">
  <img alt="version" src="https://img.shields.io/badge/version-0.1.1-blue">
</p>

- [Software description](#software-description)
- [Installation](#installation)
- [Usage Examples](#usage-example)
- [References](#literature-reference)
- [License](#license)

---
### Dependencies
**Python v3.9.6** 
```bash
pip install -r requirements.txt
```

---
## Software description

The TensorFlores framework is a Python-based solution designed for optimizing machine learning deployment in resource-constrained environments

### Software architecture

The architecture of TensorFlores can be divided into four primary layers:

- **Model Training:** A high-level API for the streamlined creation and training of MLP, supporting evolutionary vector quantization during training;

- **Json Handle:** Responsible for interpreting TensorFlow models and generating structured JSON files, serving as an intermediary representation for both TensorFlow and TensorFlores models;

- **Quantization:** Dedicated to processing the structured JSON model representation and applying PTQ techniques;

- **Code Generation:** Responsible to processing the structured representation of the JSON model and generating the machine learning model in C++ format to be embedded in the microcontroller, whether quantised or not.
    
### Software structure

The project directory is divided into key components, as illustrated in Figure:

```plaintext
tensorflores/
├── models/
│   └── multilayer_perceptron.py
├── utils/
│   ├── autocloud/
│   │   ├── auto_cloud_bias.py
│   │   ├── auto_cloud_weight.py
│   │   ├── data_cloud_bias.py
│   │   ├── data_cloud_weight.py
│   │   └── __init__.py
│   ├── array_manipulation.py
│   ├── clustering.py
│   ├── cpp_generation.py
│   ├── json_handle.py
│   ├── quantization.py
│   └── __init__.py
```

### Software functionalities

The pipeline illustrated in Figure outlines a workflow for optimizing and deploying machine learning models, specifically designed for resource-constrained environments such as microcontrollers. The software structure is divided into four main blocks: model training (with or without quantization-aware training), post-training quantization, TensorFlow model conversion, and code generation, which translates the optimized model into platform-agnostic C++ code. 

&nbsp;
<p align="center">
  <img width="800" src="https://drive.google.com/uc?export=view&id=173u4BWHWPMw0BBa4GHxRtmP1RetHZ5gD" />
</p> 
&nbsp;



The parameters are highly customizable, as shown in Table 1, which lists the class parameters and their corresponding default input values

| **Class Parameters**           | **Type** | **Input Values**                                         |
|--------------------------------|----------|----------------------------------------------------------|
| `input_size`                   | int      | 5                                                        |
| `hidden_layer_sizes`            | list     | [64, 32]                                                 |
| `output_size`                  | int      | 1                                                        |
| `activation_functions`          | list     | 'sigmoid', 'relu', 'leaky_relu', 'tanh', 'elu', 'softmax', 'softplus', 'swish', 'linear' |
| `weight_bias_init`              | str      | 'RandomNormal', 'RandomUniform', 'GlorotUniform', 'HeNormal' |
| `training_with_quantization`    | bool     | True or False                                            |

**Table 1 -** MLP Initialization Parameters.


The "train" method has the following main parameters:


| **Parameter**                 | **Type** | **Input Values**                                                                              |
|-------------------------------|----------|-----------------------------------------------------------------------------------------------|
| `X`                           | list     | List of input data for training                                                               |
| `y`                           | list     | List of corresponding labels                                                                  |
| `epochs`                      | int      | Default: 100                                                                                  |
| `learning_rate`               | float    | Default: 0.001                                                                                |
| `loss_function`               | str      | 'mean_squared_error', 'cross_entropy', 'mean_absolute_error', 'binary_cross_entropy'           |
| `optimizer`                   | str      | 'sgd', 'adam', 'adamax'                                                                      |
| `batch_size`                  | int      | Default: 36                                                                                   |
| `beta1`                       | float    | Default: 0.9 (Adam first moment)                                                              |
| `beta2`                       | float    | Default: 0.999 (Adam second moment)                                                           |
| `epsilon`                     | float    | Default: 1e-7 (Avoid division by zero in Adam)                                                |
| `epochs_quantization`         | int      | Default: 50                                                                                   |
| `distance_metric`             | str      | 'euclidean', 'manhattan', 'minkowski', 'chebyshev', 'cosine', 'hamming', 'bray_curtis', 'jaccard', 'wasserstein', 'dtw' and 'mahalanobis' |
| `bias_clustering_method`      |          | Clustering method for biases                                                                  |
| `weight_clustering_method`    |          | Clustering method for weights                                                                 |
| `validation_split`            | float    | Default: 0.2 (Validation data percentage)                                                     |

**Table 2 -** Configurable Train Method Parameters.




Table 3 presents a summary of the clustering algorithms and their respective configuration parameters.

| **Algorithm**           | **Parameter**             | **Value**  |
|-------------------------|---------------------------|------------|
| **AutoCloud**            | Threshold ($m$)           | 1.414      |
| **MeanShift**            | Bandwidth ($b$)           | 0.005      |
|                         | Maximum iterations         | 300        |
|                         | Bin seeding               | True       |
| **Affinity Propagation** | Damping ($d$)             | 0.7        |
|                         | Maximum iterations         | 500        |
|                         | Convergence iterations     | 20         |
| **DBStream**             | Clustering threshold ($\tau$) | 0.1    |
|                         | Fading factor ($\lambda$) | 0.05       |
|                         | Cleanup interval           | 4          |
|                         | Intersection factor        | 0.5        |
|                         | Minimum weight             | 1          |

**Table 3-** Clustering Algorithms and Their Respective Parameters.






## Installation

#### You can download our package from the PyPi repository using the following command:

```bash
pip install  tensorflores
```

#### If you want to install it locally you download the Wheel distribution from [Build Distribution](https://pypi.org/project/conect2py/0.1.1/#files).

*First navigate to the folder where you downloaded the file and run the following command:*

```bash
pip install tensorflores-0.1.1-py3-none-any.whl
```

---

## Usage Example

The following four examples will be considered:

### Example 01

Implementation and Training of a Neural Network Using TensorFlores:

[![Jupyter](https://img.shields.io/badge/-Notebook-191A1B?style=flat-square&logo=jupyter)](./examples/01_example_01/Example_01.ipynb)


### Example 02

Implementation and Training of a Neural Network with
quantization-aware training (QAT) Using TensorFlores:

[![Jupyter](https://img.shields.io/badge/-Notebook-191A1B?style=flat-square&logo=jupyter)](./examples/02_example_02/Example_02.ipynb)

### Example 03
Post-Training Quantization with TensorFlores:

[![Jupyter](https://img.shields.io/badge/-Notebook-191A1B?style=flat-square&logo=jupyter)](./examples/03_example_03/Example_03.ipynb)

### Example 04
Converting a TensorFlow Model using
TensorFlores: 

[![Jupyter](https://img.shields.io/badge/-Notebook-191A1B?style=flat-square&logo=jupyter)](./examples/04_example_04/Example_04.ipynb)

### Auxiliary

This section provides an example of code that transforms an input matrix (`X_test`) and (`y_test`) into a C++ array format.

[![Jupyter](https://img.shields.io/badge/-Notebook-191A1B?style=flat-square&logo=jupyter)](./examples/05_auxiliary/auxiliary.ipynb)


The Arduino code to deployment are avaliable [here](./examples/05_auxiliary/arduino_code/MLP/MLP.ino): 

## Other Models

Please check the [informations](https://github.com/conect2ai/Conect2Py-Package/blob/main/Info/AutoCloud.md) for more information about the other models been implemented in this package.


# Literature reference


1. T. K. S. Flores, M. Medeiros, M. Silva, D. G. Costa, I. Silva, Enhanced Vector Quantization for Embedded Machine Learning: A Post-Training Approach With Incremental Clustering, IEEE Access 13 (2025) 17440 17456. [doi:10.1109/ACCESS.2025.3532849](https://doi.org/10.1109/ACCESS.2025.3532849).

2. T. K. S. Flores, I. Silva, M. B. Azevedo, T. d. A. de Medeiros, M. d. A.  Medeiros, D. G. Costa, P. Ferrari, E. Sisinni, Advancing TinyMLOps:
 Robust model updates in the internet of intelligent vehicles, IEEE Micro
 (2024) [doi:10.1109/MM.2024.3354323](https://doi.org/10.1109/MM.2024.3354323).

# License

This package is licensed under the [MIT License](https://github.com/conect2ai/Conect2Py-Package/blob/main/LICENSE) - © 2023 Conect2ai.
