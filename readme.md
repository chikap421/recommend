# Book Recommender System

This project implements a book recommender system using a neural network model with PyTorch, designed to predict user book ratings. It explores different methods for data processing, including one-hot encoding and genre-based features, to prepare the input data for the neural network. The system allows for the comparison of model performance across different configurations, utilizing the mean squared error (MSE) as the performance metric.

## Features

- **PyTorch**: Utilizes PyTorch for model training, evaluation, and implementation of a fully connected neural network.
- **Neural Network**: The architecture is fully connected and allows for customizable hidden layer sizes, enabling the exploration of different model complexities.
- **Data Processing**: Incorporates functions for one-hot encoding and incorporating genre-based features, offering flexibility in how book and user information is represented.
- **Evaluation**: Employs MSE (Mean Squared Error) for evaluating the performance of the model, with support for comparing the effects of different data processing methods and network configurations.

## Installation

To set up the environment for the recommender system, follow these steps:

```bash
# Install PyTorch
Visit https://pytorch.org/get-started/locally/ for detailed installation instructions.
```

## Performance Comparison

The system compares the validation MSE across two data processing methods: one-hot encoding and genre-based features. The comparison helps in understanding which method leads to better prediction accuracy for our neural network model. Below is the comparison of validation MSE for different hidden dimensions and methods:

![Validation MSE Comparison](/plots/comparison_validation_mse.png)

This image showcases the model's performance using different configurations and data processing techniques, providing insights into the effectiveness of each approach.




## Install Pandas and NumPy
```python
pip install pandas numpy
```

## Usage
1.  Prepare your dataset in CSV format, ensuring it includes user ratings, book information, and optionally, genre information.
2.  Set the `path_to_ratings_file` variable in the main function to point to your dataset file.
3.  Adjust the `METHOD` variable in the main function to switch between 'one\_hot' and 'genre\_info' data processing methods.
4.  Run the script to train the model and evaluate its performance.