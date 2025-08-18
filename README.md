# Pain Index Prediction Modeling (CMI & VAS) Based on Clinical Data

## 1\. Project Overview

This project develops and evaluates a machine learning model to predict patient pain indices, specifically the CMI (Chronic Pain Grade) and VAS (Visual Analog Scale), using clinical data from the patient tabular data.

The core objective is to analyze which data features are more critical for predicting pain indices by using various feature sets (Full Model, Clinical + Sleep Model, and Clinical-Only Model). To achieve this, we have built a complete analysis pipeline that includes feature selection using Mutual Information, performance evaluation with a Random Forest Regressor, and visualization of the results.

## 2\. Key Features

- **Data Preprocessing**: Loads the `dataset.xlsx` file, handles missing values, converts categorical data to a numerical format, and cleans the data to make it suitable for modeling.
- **Feature Selection**: Utilizes Mutual Information (MI) with a Permutation Test to evaluate the statistical relationship between each feature and the target variables (CMI, VAS), identifying the most significant features.
- **Model Training & Evaluation**:
- Trains models using three distinct feature sets.
- Employs K-Fold Cross-Validation to ensure robust evaluation of the model's generalization performance.
- Uses MSE (Mean Squared Error) and RMSE (Root Mean Squared Error) as performance metrics.
- **Results Visualization**:
- Generates bar charts to provide an intuitive comparison of model performance across different scenarios.
- Analyzes and visualizes feature importance (MI Scores) to identify which variables are most predictive in each model.

## 3\. Requirements

The following libraries are required to run the project:

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `openpyxl` (Required by pandas to read `.xlsx` files)

## 4\. Installation

1.  **Create and activate a virtual environment** (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
.\venv\Scripts\activate  # On Windows
```

2.  **Install the required libraries**:

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm openpyxl
```

## 5\. Usage

1.  **Prepare the data**: Place the `dataset.xlsx` file in the root directory of the project.
2.  **Run the analysis**: Execute the `main.py` file from your terminal.
```bash
python main.py
```
3.  **Check the results**:
- The feature importance scores (MI Score) and p-values for each model will be printed to the terminal during execution.
- JSON files containing the model performance results (e.g., `Full_Model_model_results.json`) will be generated.
- Image files (`.png`) visualizing the model performance and feature importance will be saved and displayed on the screen.

## 6\. Project Structure

```
.
├── data/
│   └── data_preprocessing.py   # Script for data loading and preprocessing
├── utils/
│   ├── analysis_data.py        # Core analysis pipeline (MI, model evaluation)
│   ├── estimate_MI_score.py    # Functions for Mutual Information calculation
│   └── visualize_results.py    # Functions for results visualization
├── main.py                     # Main script to run the entire pipeline
└── dataset.xlsx                # (Required) The raw data file
```

### Key File Descriptions

- **`main.py`**: The entry point for the entire project. It orchestrates the workflow, starting from data preprocessing, running the analysis pipeline for the three models (A, B, C), and finally visualizing the results.
- **`data/data_preprocessing.py`**: Loads the `dataset.xlsx` file, imputes missing values using `KNNImputer`, parses date columns, converts data types, and generates a clean DataFrame ready for model training.
- **`utils/analysis_data.py`**: Contains the core analysis logic.
- `select_features_mi_with_permutation_test`: Calculates feature importance and statistical significance (p-value) using Mutual Information and a permutation test.
- `evaluate_model_with_kfold`: Evaluates the `RandomForestRegressor` model using K-Fold Cross-Validation and calculates MSE and RMSE.
- `run_analysis_pipeline`: Integrates feature selection and model evaluation to complete a single analysis scenario.
- **`utils/estimate_MI_score.py`**: Includes low-level functions using `torch` to calculate KL-Divergence and Mutual Information based on a k-Nearest Neighbors (k-NN) approach.
- **`utils/visualize_results.py`**: Uses `matplotlib` and `seaborn` to visualize the analysis results. It contains functions to generate charts for model performance comparison, error bar plots, and feature importance comparisons.
