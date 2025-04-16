# NeuroForge

This is a Flask-based web application project. The project is designed to demonstrate the use of Flask for building web applications. Below, you'll find details about the project, how to set it up, and finally how to run it on your own machine.

## Project Structure

    DMFlask/
    ├── app.py # Main Flask application file
    ├── requirements.txt # Python dependencies
    ├── Dockerfile # Docker configuration
    ├── .gitignore # Git ignore file
    ├── static/ # Static files (CSS, JS, images)
    ├── templates/ # HTML templates
    ├── models/ # Machine learning models
    │ ├── GRUModels/
    │ ├── LSTMModelsC/
    │ ├── Bi-LSTMModelsC/
    │ └── Regression/
    ├── tokenizers/ # Tokenizers for the models
    └── README.md # Project documentation

---

## Features

- Flask-based backend
- Lightweight and easy-to-deploy Docker setup
- Organized project structure for scalability
- **Machine Learning and Data Mining Features**:
  - **K-Nearest Neighbors (KNN)**: Predicts job roles based on experience and salary using Euclidean distance.
  - **K-Means Clustering**: Groups job roles and salaries into clusters for visualization.
  - **Naive Bayes (Gaussian, Multinomial, Bernoulli)**: Predicts job roles based on experience and salary.
  - **Regression**: Predicts salary based on job roles and experience.
  - **Text Generation (GRU, LSTM, Bi-LSTM)**: Generates text based on seed input using trained models.
  - **Sentiment Classification**: Classifies text as positive or negative using GRU, LSTM, or Bi-LSTM models.

---

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.10 or higher

---

## Setup Instructions

1. Clone the Repository

```bash
git clone <repository-url>
cd DMFlask
```

2. Install Dependencies
   Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask Application
   Start the Flask development server:

```bash
flask run
```

The application will be accessible at http://127.0.0.1:5000.

---

### ML & DM Features

Machine Learning and Data Mining Features

1. K-Nearest Neighbors (KNN)
   Description: Predicts job roles based on experience and salary using a custom implementation of the KNN algorithm.
   Route: /KNNPage
   Input: Experience and salary.
   Output: Predicted job role.
2. K-Means Clustering
   Description: Groups job roles and salaries into clusters for visualization.
   Route: /KMeansPage
   Input: Number of clusters.
   Output: Clustered data visualization.
3. Naive Bayes (Gaussian, Multinomial, Bernoulli)
   Description: Predicts job roles based on experience and salary using different Naive Bayes models.
   Route: /NBayesPage
   Input: Experience, salary, and Naive Bayes model type.
   Output: Predicted job role and accuracy visualization.
4. Regression
   Description: Predicts salary based on job roles and experience using a pre-trained regression model.
   Route: /RegressionPage
   Input: Job role and experience.
   Output: Predicted salary.
5. Text Generation (GRU, LSTM, Bi-LSTM)
   Description: Generates text based on a seed input using trained GRU, LSTM, or Bi-LSTM models.
   Route: /TextGenPage
   Input: Seed text, number of words to generate, and model type.
   Output: Generated text.
6. Sentiment Classification
   Description: Classifies text as positive or negative using GRU, LSTM, or Bi-LSTM models.
   Route: /ClassificationPage
   Input: Text and model type.
   Output: Sentiment classification (Positive/Negative).

---

### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Author

Developed by Rod Lester A. Moreno
