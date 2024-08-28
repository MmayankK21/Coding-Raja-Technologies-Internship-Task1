# Coding-Raja-Technologies-Internship-Task1


# Sentiment Analysis with PyTorch

This project implements a sentiment analysis model using machine learning techniques to classify text data into three sentiment categories: **Negative**, **Neutral**, and **Positive**. The model is built with PyTorch and trained on text data using TF-IDF vectorization.

## Features

- **Train Model**: Preprocess the data, vectorize text, and train a neural network model.
- **Predict Sentiment**: Input new text and predict its sentiment using the trained model.
- **Model Evaluation**: Evaluate the model on test data with metrics like accuracy, precision, recall, and F1 score.

## Project Structure

- **main.py**: Script for data preprocessing, model training, and evaluation.
- **model.py**: Contains the neural network architecture for sentiment classification.
- **predict.py**: Script to load the trained model and make sentiment predictions on new text data.

## Prerequisites

- Python 3.x
- PyTorch
- Scikit-learn
- Pandas
- Numpy
- Tqdm

Install the required packages using pip:

```bash
pip install torch scikit-learn pandas numpy tqdm
```

## Dataset

The dataset used for training is `twitter_training.csv`. Make sure the dataset is correctly formatted and placed in the appropriate path defined in `main.py`.

## How to Use

### 1. Train the Model

To train the model, run the following command:

```bash
python main.py
```

This script will:
- Load and preprocess the dataset.
- Split the dataset into training and test sets.
- Vectorize the text data using a TF-IDF vectorizer.
- Train a feedforward neural network using PyTorch.
- Save the trained model (`sentiment_model.pth`) and vectorizer (`vectorizer.pkl`).

### 2. Predict Sentiment

To predict the sentiment of new text, run:

```bash
python predict.py
```

You will be prompted to input text, and the model will return the sentiment as **Negative**, **Neutral**, or **Positive**.

Example:
```
Enter text to predict sentiment (or 'quit' to exit): This product is amazing!
Sentiment: Positive
```

### 3. Model Architecture

The `SentimentModel` is a simple feedforward neural network with the following layers:
- **Input layer**: Takes the TF-IDF features (with 10,000 dimensions).
- **Hidden layer**: Fully connected with 512 units and ReLU activation.
- **Dropout layer**: Adds regularization to prevent overfitting.
- **Output layer**: Outputs probabilities for three sentiment classes (Negative, Neutral, Positive).

### 4. Evaluation

After training, the model is evaluated on the test data. The following metrics are displayed:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

## Example Output

Sample training output:

```
Training epoch 1/10...
Accuracy: 0.85, Precision: 0.83, Recall: 0.84, F1 Score: 0.84
```

Sample prediction:

```
Enter text to predict sentiment (or 'quit' to exit): I love this movie!
Sentiment: Positive
```

## Author

[Mayank Wadhwa](https://github.com/MmayankK21)

