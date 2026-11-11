# CPU Performance Prediction Using Machine Learning

## Group - 61

## 1. Project Overview

This project focuses on predicting CPU performance using Machine Learning techniques in Python.

The project uses CPU hardware specifications as input features to predict the estimated performance of a CPU. It involves data preprocessing, exploratory data analysis, data visualization, model training and model evaluation.

---

## 2. Team Members and Roles

The project is divided among four team members, with each member responsible for a specific component.

| Team Member | Name | entry No. | Role |
|---|---|---|---|
| Member 1 | Prabhnoor Singh | 2026AIB1067 | Execution Lead |
| Member 2 | Ansh Raj | 2026EEB1323 | Video Lead |
| Member 3 | Avani Goutam | 2026EEB1329 | PPT Lead |
| Member 4 | Karan Kumar | 2026MCB1533 | documentation Lead |

All team members contributed to the project through coding, testing and GitHub collaboration.

---

## 3. Project Structure

The repository is organized into the following directories and files:

```text
CPU-Performance-Prediction/
│
├── data/
│   └── Sample-Data-CPU-Performance.csv
│
├── notebooks/
│   ├── data_preparation.py
│   ├── data_visualisation.py
│   ├── model_preparation.py
│   └── model_evaluation.py
│
├── outputs/
│   ├── cleaned_dataset.csv
│   ├── features.csv
│   ├── target.csv
│   ├── correlation_heatmap.png
│   ├── feature_relationships.png
│   ├── trained_model.pkl
│   ├── evaluation_metrics.csv
│   └── actual_vs_predicted.png
│
├── main.py
├── requirements.txt
└── README.md
```

**Directory description:**

- `data/` – Contains the original CPU performance dataset.
- `notebooks/` – Contains the Python scripts for the four project components.
- `outputs/` – Stores the processed datasets, visualizations, trained model and evaluation results.
- `main.py` – Executes all four project components sequentially.
- `requirements.txt` – Contains the required Python libraries.
- `README.md` – Provides project documentation and execution instructions.

---

## 4. Dataset Description

**Dataset:** `Sample-Data-CPU-Performance.csv`

The dataset contains CPU specifications that are used to predict CPU performance.

**Input Features:**

1. CycleTime
2. MinimumMainMemory
3. MaximumMainMemory
4. CacheSize
5. MinimumNumberOfChannels
6. MaximumNumberOfChannels

**Target Variable:**

- `EstimatedPerformance`

The project uses a regression approach to estimate CPU performance based on the given hardware specifications.

---

## 5. Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 6. Setup and Installation

### Prerequisites

- Python [3.14.6]
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/prabhnoorprnc/AI101-Project-.git
```

### Step 2: Navigate to the Project Directory

```bash
cd AI101-Project-
```

### Step 3: Create a Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## 7. How to Run the Project

After completing the installation, make sure you are in the main project directory.

Execute the following command:

```bash
python main.py
```

The `main.py` file automatically executes all four project components in the correct sequence:

1. **Data Preprocessing:** Cleans the dataset and prepares the features and target variable.
2. **Data Analysis and Visualization:** Performs exploratory data analysis and generates visualizations.
3. **Model Training:** Trains the machine learning regression model.
4. **Model Evaluation:** Evaluates the trained model and generates performance results.

The generated datasets, graphs, trained model and evaluation results are saved in the `outputs/` directory.

---

## 8. Expected Outputs

After successful execution, the following outputs are generated:

| Output | Description |
|---|---|
| cleaned_dataset.csv | Preprocessed dataset |
| features.csv | Input features |
| target.csv | Target variable |
| correlation_heatmap.png | Correlation between features |
| feature_relationships.png | Feature relationship visualizations |
| trained_model.pkl | Saved trained model |
| evaluation_metrics.csv | Model evaluation metrics |
| actual_vs_predicted.png | Comparison of actual and predicted values |


## 9. Presentation and Video

### Project Presentation

The project presentation contains the problem statement, dataset description, methodology, results and key learnings.

**PPT Link:** [Insert Google Drive or GitHub PPT link]

### Project Demonstration Video

The demonstration video explains the project workflow, code execution, results and individual team contributions.

**Video Link:** [Insert YouTube or Google Drive video link]

---

## 10. AI Use Disclosure

AI tools were used as supporting resources during the development of this project.

- **ChatGPT :** Assisted with improving code structure , making readme structure and making overall structure of PPT.
- The team members are responsible for understanding the implementation, code, methodology and results of their respective project components.

---
