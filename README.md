Product Recommendation Model - Task 1
This project is part of the "User Identity and Product Recommendation System". The primary goal of this task is to process and merge customer data to train a machine learning model that predicts which product category a user is likely to be interested in based on their social media activity.


⚙️ Setup and Installation
Follow these steps to set up the project environment and install all necessary dependencies.

1. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage project-specific dependencies. Navigate to the project's root directory and run:

Bash

python3 -m venv venv
2. Activate the Environment
On macOS/Linux:

Bash

source venv/bin/activate
On Windows:

Bash

venv\Scripts\activate
3. Create and Use requirements.txt
A requirements.txt file lists all the Python libraries the project needs.

To create the file (Owner's task):
After installing all necessary packages (like pandas, scikit-learn, numpy), run the following command to save them to a file:

Bash

pip freeze > requirements.txt
To install from the file (User's task):
Anyone who clones this repository can install all the required dependencies with a single command:

Bash

pip install -r requirements.txt
📊 Model Performance Explained
The model's performance is evaluated using several key metrics from the final classification report.

Accuracy: This is the most straightforward metric. It measures the percentage of predictions the model got right. An accuracy of 0.77 means the model was correct 77% of the time.

Precision: This answers the question: "Of all the times the model predicted a specific category, how often was it correct?" High precision means the model is reliable in its positive predictions.

Recall: This answers the question: "Of all the actual instances of a category, how many did the model successfully identify?" High recall means the model is good at finding all instances of a category.

F1-Score: This is the harmonic mean of Precision and Recall. It provides a single score that balances both metrics, which is especially useful when there is a class imbalance. An F1-Score of 0.74 indicates a strong balance between precision and recall.

📈 Results
The initial datasets were found to have no common customers, making a direct merge impossible. To proceed, a simulated dataset was generated with intentional, logical patterns connecting user features to product interests.

After feature engineering and hyperparameter tuning, the final RandomForestClassifier achieved:

Accuracy: 0.77

Weighted F1-Score: 0.74

These strong results demonstrate that the model successfully learned the engineered patterns from the simulated data.