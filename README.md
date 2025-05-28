# Machine Learning Data Preprocessor

A versatile CLI tool that transforms raw datasets into ML-ready features.  
Pick your model type and configure:
- **Target column** to predict (optional)  
- **Columns to drop** (optional)  
- **Model-specific preprocessing** (Tree/Ensemble, Linear/SVM, or KNN/Deep)

## Features

- **Imputation**: Mean, median, or most frequent for missing values  
- **Scaling**: Standard or MinMax based on model choice  
- **Encoding**: One-hot encoding with safe handling of unknown categories  
- **Outlier handling**: IQR-based clipping for robust linear/SVM and KNN models  

*Future versions* will add advanced encoding options, feature engineering, and more.

## Requirements

- Python 3.8+
- pandas
- scikit-learn

## Usage

```bash
python preprocess.py --path data.csv --model 2 --target price --ignore-cols id,date


**Note**: This is my first project uploaded to GitHub. Feedback and suggestions are welcome. I'm looking forward to any suggestions and advice to further improve my project.

