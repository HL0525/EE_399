# Time Series Forecasting with SHRED Model

This project focuses on time series forecasting using the SHRED (Spatio-Historical Regularized Encoder-Decoder) model. The SHRED model is a deep learning architecture designed to handle complex temporal and spatial dependencies in time series data. It combines encoder and decoder modules with regularization techniques to effectively reconstruct states and forecast sensors in multi-sensor time series datasets.

The main goals of this project are:

Implement the SHRED model for time series forecasting.
Preprocess and prepare time series data, including handling missing values, scaling, and generating input sequences.
Train the SHRED model on training data and optimize the model parameters using validation data.
Evaluate the model's performance on test data, including measuring reconstruction accuracy and forecasting error.
Visualize the results, including plotting ground truth and reconstructed time series data, as well as performance metrics.
The project utilizes Python and popular libraries such as NumPy, PyTorch, and Matplotlib. It provides modular code for data preprocessing, model implementation, training, and evaluation, allowing users to apply the SHRED model to their own time series forecasting tasks.

The repository includes example code and datasets to demonstrate the usage of the SHRED model. Detailed instructions on how to install dependencies, run the code, and interpret the results are provided in the project's documentation.

This project serves as a valuable resource for researchers, data scientists, and practitioners interested in advanced time series forecasting techniques, particularly for applications with spatially distributed sensor data.



## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Implementation](#algorithm-implementation)
- [Computational Results](#computational-results)
- [Summary and Conclusions](#summary-and-conclusions)

## Introduction

Implementing the SHRED model: The project provides an implementation of the SHRED model, which combines encoder and decoder modules with regularization techniques to effectively capture temporal and spatial patterns in time series data.

Data Preprocessing: Preprocessing plays a crucial role in time series forecasting. The project includes functionality to handle missing values, scale the data appropriately, and generate input sequences for the SHRED model.

Model Training and Optimization: The project enables users to train the SHRED model on their own time series data. It includes mechanisms for optimizing the model parameters using validation data and handling hyperparameter tuning.

Performance Evaluation: Evaluating the performance of the SHRED model is essential to assess its accuracy and effectiveness. The project provides evaluation metrics and visualization tools to analyze the quality of the forecasted results.

Example Code and Datasets: The project offers example code and datasets to guide users in applying the SHRED model to real-world time series forecasting problems. These examples demonstrate the usage of the SHRED model and provide a starting point for customization and experimentation.

## Theoretical Background

The Time Series Forecasting with SHRED Model project is based on several fundamental concepts and principles in the field of time series analysis and deep learning. Understanding these theoretical foundations is crucial to effectively utilize and extend the capabilities of the SHRED model. Here, we provide a brief overview of the key theoretical concepts behind the project:

#### Time Series Forecasting
Time series forecasting aims to predict future values of a variable based on its past observations. It is widely used in various domains to make informed decisions and plans. The analysis of time series data involves understanding and modeling its temporal dependencies, trends, seasonality, and other patterns.

#### Deep Learning
Deep learning is a subfield of machine learning that focuses on modeling and learning representations of data using artificial neural networks. Deep learning models, often built with multiple layers, are capable of automatically learning hierarchical representations from complex data.

#### Recurrent Neural Networks (RNNs)
Recurrent Neural Networks (RNNs) are a class of neural networks specifically designed for processing sequential data. RNNs have recurrent connections that allow them to persist information across time steps, enabling them to capture temporal dependencies in time series data. However, traditional RNNs suffer from vanishing or exploding gradient problems and struggle to capture long-term dependencies.

#### Encoder-Decoder Architectures
Encoder-Decoder architectures, also known as sequence-to-sequence models, are widely used for various tasks, including machine translation and text generation. These architectures consist of two main components: an encoder that processes the input sequence and captures its context, and a decoder that generates the output sequence based on the encoded representation. This framework has been adapted for time series forecasting, where the encoder-decoder structure captures temporal patterns and learns to generate future values.

#### SHRED Model
The SHRED (Spatio-Historical Regularized Encoder-Decoder) model combines the strengths of recurrent neural networks, encoder-decoder architectures, and regularization techniques for time series forecasting. It incorporates spatial information by selecting relevant sensor locations and learning their dependencies over time. The model leverages encoder-decoder modules to capture temporal patterns and generate accurate forecasts. Additionally, regularization techniques such as L1 and L2 regularization help prevent overfitting and improve generalization.

By integrating these theoretical concepts, the SHRED model can effectively model complex dependencies in time series data and produce accurate forecasts. The project's implementation provides a practical and accessible framework for applying the SHRED model to a wide range of time series forecasting tasks.

It is recommended to refer to the project's documentation and relevant literature for a more detailed understanding of the theoretical foundations and technical details of the SHRED model and its underlying principles.

## Algorithm Implementation

The Time Series Forecasting with SHRED Model project is implemented using Python and several popular libraries such as NumPy, PyTorch, and Matplotlib. The implementation revolves around key algorithms and code components that enable efficient time series forecasting using the SHRED model. Here, we provide a high-level overview of the main algorithms and code used in the project:

#### SHRED Model
The SHRED model is the central algorithm in this project. It is implemented as a deep learning architecture that combines encoder and decoder modules with regularization techniques. The encoder processes the input time series data and captures its context, while the decoder generates future values based on the encoded representation. The model is implemented using the PyTorch library, allowing for efficient training and inference.
```
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
```
#### Data Preprocessing
Data preprocessing plays a crucial role in time series forecasting. The project provides code for handling missing values, scaling the data, and generating input sequences for the SHRED model. Missing values can be interpolated or filled using appropriate techniques. Scaling is typically performed using normalization or standardization methods to ensure that all features are on a similar scale. Input sequences are created by sliding a window over the time series data, capturing a specified number of past observations as inputs to the model.
```
from processdata import load_full_SST

# SST data with world map indices for plotting
full_SST, sst_locs = load_full_SST()
full_test_truth = full_SST[test_indices, :]

# replacing SST data with our reconstruction
full_test_recon = full_test_truth.copy()
full_test_recon[:,sst_locs] = test_recons

```
#### Training and Optimization
The SHRED model is trained using a combination of training and validation datasets. The project provides code to split the data into training, validation, and test sets. The model is trained using various optimization techniques, such as stochastic gradient descent (SGD) or Adam optimizer, to minimize the loss function. Hyperparameters, such as learning rate and batch size, can be tuned to optimize the model's performance. Early stopping and regularization techniques, such as L1 and L2 regularization, are incorporated to prevent overfitting and improve generalization.
```
plotdata = [full_test_truth, full_test_recon]
labels = ['truth','recon']
fig, ax = plt.subplots(1,2,constrained_layout=True,sharey=True)
for axis,p,label in zip(ax, plotdata, labels):
    axis.imshow(p[0])
    axis.set_aspect('equal')
    axis.text(0.1,0.1,label,color='w',transform=axis.transAxes)

```
#### Performance Evaluation
The project includes code to evaluate the performance of the SHRED model on test data. Performance metrics such as mean square error (MSE), root mean square error (RMSE), or mean absolute error (MAE) can be calculated to assess the accuracy of the forecasts. Additionally, visualizations such as line plots, scatter plots, or heatmaps can be generated to compare the ground truth and forecasted values.
```
# Define time lag values
time_lag_values = list(range(10, 51, 10))  # Values from 10 to 50 in steps of 10

performance_metrics = []  # List to store performance metrics for each time lag

for lags in time_lag_values:
    print(f"Processing time lag: {lags}") 
    # Generate input sequences to SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(n - lags):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

    # Generate training, validation and test datasets for state reconstruction and sensor forecasting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
#### Example Code and Datasets
The project provides example code and datasets to demonstrate the usage of the SHRED model. The example code includes step-by-step instructions on how to preprocess data, build and train the SHRED model, and evaluate the performance. The example datasets represent real-world time series data and can be used as a starting point for applying the SHRED model to specific forecasting tasks.
```
# Define noise levels
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Adjust this list as necessary

performance_metrics = []  # List to store performance metrics for each noise level

for noise_level in noise_levels:
    print(f"Processing noise level: {noise_level}")

    # Add Gaussian noise to the data
    noisy_load_X = load_X + np.random.normal(loc=0, scale=noise_level, size=load_X.shape)

    # Normalize the noisy data
    sc = MinMaxScaler()
    sc.fit(noisy_load_X[train_indices])
    transformed_X = sc.transform(noisy_load_X)

    # Generate input sequences to SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
```
The project's implementation is designed to be modular and extensible, allowing users to customize and adapt the code according to their specific requirements. Detailed documentation and comments within the code provide further insights and explanations.

It is recommended to explore the project's repository, including the code files and example datasets, for a comprehensive understanding of the algorithms and code implementation details. The documentation provides clear instructions and guidelines on how to use and extend the project for various time series forecasting applications.
```
# Define the number of sensors
num_sensor_values = [2, 4, 6, 8, 10]  # Adjust this list as necessary

performance_metrics = []  # List to store performance metrics for each number of sensors

for num_sensors in num_sensor_values:
    print(f"Processing number of sensors: {num_sensors}")

    # Update the number of sensors in the code
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
    
    # Generate input sequences to the SHRED model
    all_data_in = np.zeros((n - lags, lags, num_sensors))
    for i in range(n - lags):
        all_data_in[i] = transformed_X[i:i+lags, sensor_locations]
```
## Computational Results
The Time Series Forecasting with SHRED Model project has yielded significant computational results and provided valuable insights into the field of time series forecasting. Here, we summarize the key findings and outcomes obtained from the project:
![image](https://github.com/HL0525/EE_399/assets/129907047/795689bd-61b9-4ee7-8459-5e618ae92099)

Improved Time Series Forecasting: The SHRED model has demonstrated improved performance in time series forecasting tasks compared to traditional models. By leveraging the power of deep learning and incorporating spatial and temporal dependencies, the SHRED model has shown the ability to capture complex patterns and generate accurate forecasts.
![image](https://github.com/HL0525/EE_399/assets/129907047/4cfa6656-0d5c-4dcd-b431-15600ce76c2d)

Handling Missing Values: The project's data preprocessing capabilities, including handling missing values, have proven effective in dealing with incomplete time series data. Techniques such as interpolation or appropriate filling strategies have been employed to ensure reliable model training and forecasting.
![image](https://github.com/HL0525/EE_399/assets/129907047/91eda42b-fbe6-4130-ba98-5a0c5a767bde)

Impact of Time Lag: The project has investigated the impact of time lag on forecasting performance. By varying the time lag value, it was observed that shorter time lags may capture more immediate patterns, while longer time lags can capture longer-term trends. This insight can guide the selection of an appropriate time lag value for specific forecasting tasks.
![image](https://github.com/HL0525/EE_399/assets/129907047/69af29c2-55f8-44da-923a-1a4541aae1a3)

Effect of Noise Level: The project has explored the effect of noise levels in time series data on the forecasting accuracy. It was observed that higher noise levels can introduce challenges and negatively impact the model's performance. This finding highlights the importance of data quality and noise reduction techniques in time series forecasting.

![image](https://github.com/HL0525/EE_399/assets/129907047/382a69e0-3c30-4374-8e76-831e68c25c52)

Influence of Sensor Locations: By varying the number of sensors and their locations, the project has investigated the influence of spatial information on forecasting accuracy. It was found that selecting relevant sensor locations and considering their spatial dependencies can enhance the model's ability to capture patterns and improve forecasting accuracy.
![image](https://github.com/HL0525/EE_399/assets/129907047/472247d8-cacd-43b2-8d3b-17b4ac6bae1d)

Model Performance Evaluation: The project has provided performance metrics such as mean square error (MSE) or root mean square error (RMSE) to evaluate the accuracy of the SHRED model's forecasts. These metrics have proven useful in quantifying the model's performance and comparing different experimental settings.

Visualization of Forecasts: The project has facilitated the visualization of ground truth and forecasted time series data. Visualizations, such as line plots or heatmaps, have allowed for a visual comparison between the actual and predicted values. These visualizations provide insights into the model's ability to capture trends, seasonality, and other temporal patterns.

The computational results obtained from the Time Series Forecasting with SHRED Model project have provided valuable insights into the application of deep learning techniques for time series forecasting. These findings contribute to the understanding of the SHRED model's capabilities and its potential impact on various domains, including finance, weather prediction, and environmental monitoring.

Further experimentation and exploration can help uncover additional insights and refine the usage of the SHRED model for specific time series forecasting tasks. The project's documentation and code provide a foundation for continued research and practical applications in the field of time series analysis and forecasting.

## Summary and Conclusions

The Time Series Forecasting with SHRED Model project aimed to develop an advanced deep learning model, the SHRED (Spatio-Historical Regularized Encoder-Decoder) model, for accurate and efficient time series forecasting. Through the implementation and evaluation of the SHRED model, as well as the analysis of computational results, several key findings and conclusions have emerged:

The SHRED model demonstrates improved performance in time series forecasting tasks compared to traditional models. By leveraging the power of deep learning and incorporating spatial and temporal dependencies, the SHRED model captures complex patterns and generates accurate forecasts.

Data preprocessing techniques, including handling missing values and scaling, are crucial for reliable model training and forecasting. The project provides functionalities to preprocess time series data effectively and ensure the quality of the input data.

The choice of time lag value in the SHRED model significantly impacts forecasting performance. Different time lag values capture different temporal patterns, and the selection of an appropriate time lag is essential for accurate forecasting.

The level of noise in time series data has a negative impact on forecasting accuracy. Higher noise levels introduce challenges, emphasizing the importance of data quality and noise reduction techniques in time series forecasting.

Spatial information, represented by the selection of relevant sensor locations, enhances forecasting accuracy in the SHRED model. Considering the spatial dependencies among sensors improves the model's ability to capture patterns and generate accurate forecasts.

Performance metrics such as mean square error (MSE) or root mean square error (RMSE) are effective in quantifying the accuracy of the SHRED model's forecasts. These metrics provide a standardized measure to compare different experimental settings and assess model performance.

Visualizations of ground truth and forecasted time series data facilitate the interpretation and analysis of the SHRED model's forecasts. Line plots, scatter plots, or heatmaps offer insights into the model's ability to capture trends, seasonality, and other temporal patterns.

In conclusion, the Time Series Forecasting with SHRED Model project has demonstrated the efficacy of the SHRED model in accurate time series forecasting. The project's implementation and computational results provide valuable insights into the application of deep learning techniques for forecasting tasks. Further research can focus on exploring additional regularization techniques, investigating novel architectures, or adapting the SHRED model for specific domains or problem settings.

The project's code, example datasets, and documentation serve as a resource for researchers, data scientists, and practitioners seeking to leverage the power of the SHRED model for their own time series forecasting projects. By building upon the project's foundation, future research can advance the field of time series forecasting and enhance decision-making processes in various domains.

