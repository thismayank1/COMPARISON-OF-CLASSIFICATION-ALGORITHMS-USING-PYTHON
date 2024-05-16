# algo/views.py
from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Import Matplotlib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
import os

# Assuming the 'diabetes.csv' file is in the same directory as your views.py


def index(request):
    # Your existing code to load and preprocess data
    csv_path = os.path.join(os.path.dirname(__file__), 'diabetes.csv')
    data = pd.read_csv(csv_path)

    x = np.array(data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]])
    y = np.array(data[["Outcome"]])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

    # Initialize classifiers
    classifiers = {
        'KNN Classifier': KNeighborsClassifier(),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Passive Aggressive Classifier': PassiveAggressiveClassifier()
    }

    results = []

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        clf.fit(xtrain, ytrain)
        y_pred = clf.predict(xtest)

        # Generate and append classification report to results
        report = classification_report(ytest, y_pred, output_dict=True)
        results.append({'name': name, 'report': report})

    # Additional code to display the accuracy scores
    data1 = {"Classification Algorithms": list(classifiers.keys()),
            "Score": [clf.score(x, y) for clf in classifiers.values()]}

    score = pd.DataFrame(data1)

    # Create a bar chart using Matplotlib
    plt.bar(data1["Classification Algorithms"], data1["Score"])
    plt.xlabel('Classification Algorithms')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Scores of Classification Algorithms')
    plt.ylim([0, 1])  # Set the y-axis limit to 0-1 for accuracy score
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    # Save the plot to a file or render it directly in the template
    plt.savefig(os.path.join(os.path.dirname(__file__), 'static', 'accuracy_chart.png'))  # Save the plot as an image file
    plt.close()  # Close the plot to free up resources

    # Render the results and the chart in a template
    return render(request, 'index.html', {'results': results, 'score': score.to_html()})
