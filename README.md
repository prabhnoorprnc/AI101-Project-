# CPU Performance Prediction Using Machine Learning

A Data Science and Artificial Intelligence mini-project that uses Python and machine learning techniques to analyse CPU performance data and predict estimated CPU performance based on hardware-related features.

---

## 1. Project Overview

This project focuses on analysing CPU performance data and developing a machine learning model to predict the estimated performance of a CPU.

The project is divided into four main stages:

1. **Data Preprocessing:** Cleaning the dataset, extracting relevant numerical values, and preparing features and target variables.
2. **Data Analysis and Visualisation:** Exploring the dataset using statistical analysis and graphical visualisations.
3. **Model Training:** Preparing the data and training a machine learning model.
4. **Model Evaluation:** Evaluating the trained model using suitable performance metrics and visualisations.

Each team member is responsible for one stage of the project. All members contribute code and maintain their work through GitHub commits.

## 2. Project Objectives

- Understand and preprocess CPU performance data.
- Identify important features related to CPU performance.
- Perform exploratory data analysis and visualisation.
- Train a machine learning model to estimate CPU performance.
- Evaluate the model using appropriate evaluation metrics.
- Collaborate using GitHub for version control and project management.

## 3. Dataset Description

**Dataset name:** `Sample-Data-CPU-Performance.csv`

The dataset contains CPU-related specifications and their corresponding estimated performance values.

### Features used

| Feature | Description |
|---|---|
| CycleTime | CPU cycle time |
| MinimumMainMemory | Minimum main memory |
| MaximumMainMemory | Maximum main memory |
| CacheSize | Cache size |
| MinimumNumberOfChannels | Minimum number of channels |
| MaximumNumberOfChannels | Maximum number of channels |

**Target variable:** `EstimatedPerformance`

The dataset is included in the `data/` directory of this repository.

## 4. Team Members and Responsibilities

Team Member     | Entry No.   | Roles

Prabhnoor Singh | 2026AIB1067 | Execution Lead
Ansh Raj        | 2026EEB1323 | Video Lead 
Avani Goutam    | 2026EEB1329 | PPT Lead
Karan Kumar     | 2026MCB1533 | Documentation Lead

All four members contribute to coding and commit their work using their own GitHub accounts.

## 5. Project Structure

text
AI101-Project-/
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
│   ├── Data_Processing/
│   ├── Data_Analysis_Visualization/
│   ├── Model_Training/
│   └── Model_Evaluation/
│
├── README.md
└── requirements.txt


**Folder descriptions:**

- `data/` – Contains the original dataset.
- `notebooks/` – Contains Python scripts for all four project stages.
- `outputs/Data_Processing/` – Stores cleaned data, features, and target files.
- `outputs/Data_Analysis_Visualization/` – Stores generated graphs and visualisation results.
- `outputs/Model_Training/` – Stores model training outputs.
- `outputs/Model_Evaluation/` – Stores model evaluation results.
- `requirements.txt` – Contains the Python libraries required to run the project.

## 6. Requirements

Before running the project, ensure that the following software is installed:

- Python (version used and tested by the team: **[3.14.6]**)
- Git
- pip (Python package installer)

All required Python libraries are listed in `requirements.txt`.

## 7. Installation and Setup

Follow these steps to download and run the project on a new Windows computer.

### Step 1: Clone the repository

Open PowerShell and execute:

```powershell
git clone https://github.com/prabhnoorprnc/AI101-Project-.git
```

### Step 2: Navigate to the project directory

```powershell
cd AI101-Project-
```

### Step 3: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 4: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### Step 5: Install the required libraries

```powershell
python -m pip install -r requirements.txt
```

After installation, the project is ready to run.

## 8. Steps to Execute the Project

**Important:** Run the scripts in the following order because later stages depend on the outputs of earlier stages.

### 1: Data Preprocessing

```powershell
python .\notebooks\data_preparation.py
```

This stage:
- Loads the original CPU performance dataset.
- Extracts and prepares the required numerical features.
- Cleans and organises the dataset.
- Separates the input features (`X`) and target variable (`y`).
- Saves the processed files in `outputs/Data_Processing/`.

Expected files:

- `cleaned_dataset.csv`
- `features.csv`
- `target.csv`

### 2: Data Analysis and Visualisation

```powershell
python .\notebooks\data_visualisation.py
```

This stage:
- Loads the cleaned dataset.
- Performs exploratory data analysis.
- Generates visualisations to understand feature distributions and relationships.
- Saves the generated graphs in `outputs/Data_Analysis_Visualization/`.

### 3: Model Training

```powershell
python .\notebooks\model_preparation.py
```

This stage:
- Loads the prepared features and target variable.
- Prepares the data for model training.
- Splits the dataset into training and testing sets, where applicable.
- Trains the selected machine learning model.
- Saves the relevant model training outputs in `outputs/Model_Training/`.

### 4: Model Evaluation

```powershell
python .\notebooks\model_evaluation.py
```

This stage:
- Evaluates the trained machine learning model.
- Calculates suitable evaluation metrics.
- Compares actual and predicted values.
- Generates relevant evaluation visualisations.
- Saves the evaluation results in `outputs/Model_Evaluation/`.

## 9. Expected Outputs

After successfully executing all four stages, the `outputs/` directory should contain the generated files from preprocessing, visualisation, model training, and model evaluation.

To inspect the generated files, run:

```powershell
Get-ChildItem .\outputs -Recurse
```

### Expected Output Screenshot

Add a screenshot showing the successful execution of the project and its important results.

![Expected project output](outputs/screenshots/expected_output.png)

*Replace the image path above with the actual screenshot path committed to the repository.*

## 10. Presentation Slides

The project presentation contains exactly five slides, covering:

1. Title and team introduction
2. Problem statement and dataset
3. Approach and methodology
4. Results
5. Learnings, contributions, and AI-use disclosure

**Presentation:** [Add link to PPT file or shared presentation]

## 11. Project Demonstration Video

The project demonstration video explains the project workflow, implementation, execution, results, and individual contributions.

The video includes a demonstration of cloning the GitHub repository, installing the required libraries, and running the project.

**Video link:** [Add publicly accessible video link]

*The demonstration video is within the maximum duration of five minutes.*

## 12. AI-Use Disclosure

AI tools were used as supporting resources during project development.

- **ChatGPT:** Assisted with understanding Python code, debugging, project organisation, execution instructions, and drafting documentation.
- All AI-assisted suggestions are reviewed and adapted by the team.
- Team members are responsible for understanding the code, implementation, and results they submit.

[Add details of any other AI tools used by the team, or state that no other tools were used.]

## 13. Conclusion

This project provides practical experience in the complete machine learning workflow, from data preprocessing and exploratory data analysis to model training and evaluation.

It also demonstrates teamwork, Python programming, data visualisation, machine learning implementation, and collaborative development using GitHub.

---

**Course:** AI101 – Data Science and AI Essentials with Python  
**Institution:** School of AI and Data Engineering, IIT Ropar
