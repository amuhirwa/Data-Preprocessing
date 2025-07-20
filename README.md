# Product Recommendation Model

[cite\_start]This project involves processing and merging customer data from social media profiles and transaction histories to train a machine learning model[cite: 42, 43, 44]. [cite\_start]The model's goal is to predict which product a customer is likely to purchase[cite: 45].

-----

## ⚙️ Setup and Installation

To get this project running on your local machine, follow these steps.

### 1\. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2\. Create and Activate a Virtual Environment

It is a best practice to create a virtual environment to manage project-specific dependencies.

```bash
# Create the environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3\. Install Dependencies

All the required Python libraries are listed in the `requirements.txt` file. Install them with a single command:

```bash
pip install -r requirements.txt
```

This file will contain the necessary libraries such as `pandas`, `scikit-learn`, and `numpy`.

-----

## 🚀 Running the Project

The core logic for data merging, model training, and evaluation is contained within the Jupyter Notebook. Once your environment is set up and dependencies are installed, you can start Jupyter Lab:

```bash
jupyter lab
```

From the Jupyter interface in your browser, open the main notebook file (e.g., `Formative_2_Data_Preprocessing.ipynb`) and run the cells in order from top to bottom.
