import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def column_preprocessing(dataframe, numerical_column, categorical_column):  # GENERAL COLUMN PREPROCESSING
    most_frequent = [
        col for col in categorical_column
        if (len(dataframe) / dataframe[col].nunique(dropna=True)) > 2
    ]

    low_cardinality_categorical = [
        col for col in categorical_column
        if (len(dataframe) / dataframe[col].nunique(dropna=True)) <= 2
        # If more than 50% of the total size of the row has unique values, high cardinality.
    ]
    mean_numeric = [
        col for col in numerical_column
        if -0.5 <= dataframe[col].skew() <= 0.5
    ]
    median_numeric = [
        col for col in numerical_column
        if dataframe[col].skew() <= -0.5 or dataframe[col].skew() >= 0.5
    ]

    return most_frequent, low_cardinality_categorical, mean_numeric, median_numeric


def general_preprocessing(model, dataframe, features):  # MAIN PREPROCESSOR
    numerical_column = [column for column in features if features[column].dtype in ["int64", "float64"]]
    categorical_column = [column for column in features if features[column].dtype == "object"]
    for column in categorical_column:
        dataframe[column] = dataframe[column].str.upper()

    if model == 1:  # TREE / ENSEMBLE
        most_frequent, low_cardinality_categorical, mean_numeric, median_numeric = column_preprocessing(dataframe,
                                                                                                        numerical_column,
                                                                                                        categorical_column)

        most_frequent_transform = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        low_cardinality_categorical_transform = make_pipeline(SimpleImputer(strategy="constant", fill_value="MISSING"),
                                                              OneHotEncoder(handle_unknown="ignore",
                                                                            sparse_output=True))
        mean_numeric_transform = make_pipeline(SimpleImputer(strategy="mean"))
        median_numeric_transform = make_pipeline(SimpleImputer(strategy="median"))

    elif model == 2:  # LINEAR / SVM
        df = dataframe.copy()
        for column in numerical_column:
            q1 = df[column].quantile(0.25)

            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

        most_frequent, low_cardinality_categorical, mean_numeric, median_numeric = column_preprocessing(df,
                                                                                                        numerical_column,
                                                                                                        categorical_column)

        most_frequent_transform = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        low_cardinality_categorical_transform = make_pipeline(SimpleImputer(strategy="constant", fill_value="MISSING"),
                                                              OneHotEncoder(handle_unknown="ignore",
                                                                            sparse_output=True))
        mean_numeric_transform = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler())
        median_numeric_transform = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    elif model == 3:  # KNN / DEEP
        df = dataframe.copy()
        for column in numerical_column:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

        most_frequent, low_cardinality_categorical, mean_numeric, median_numeric = column_preprocessing(df,
                                                                                                        numerical_column,
                                                                                                        categorical_column)

        most_frequent_transform = make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        low_cardinality_categorical_transform = make_pipeline(SimpleImputer(strategy="constant", fill_value="MISSING"),
                                                              OneHotEncoder(handle_unknown="ignore",
                                                                            sparse_output=True))
        mean_numeric_transform = make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler())
        median_numeric_transform = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())

    preprocessor = ColumnTransformer([
        ("most_frequent", most_frequent_transform, most_frequent),
        ("low_cardinality_categorical_transform", low_cardinality_categorical_transform, low_cardinality_categorical),
        ("mean_numeric", mean_numeric_transform, mean_numeric),
        ("median_numeric", median_numeric_transform, median_numeric)
    ])

    preprocessor.fit(features)
    return preprocessor

path = r""  # Path to the CSV file (include full directory and filename)

try:
    dataframe = pd.read_csv(path)
except FileNotFoundError:
    print("Error: File not found")
    sys.exit(1)

dataframe_copy = dataframe.copy()
dataframe_copy = dataframe_copy.dropna(thresh=len(dataframe) * 0.5, axis=1)

while True:  # IGNORED COLUMNS
    try:
        ignored_columns = input("Enter columns to ignore (Comma-seperated, 0 if none): ")
        if ignored_columns.strip() != "0":
            ignored_columns = [col.strip() for col in ignored_columns.split(",")]
            dataframe_copy = dataframe_copy.drop(ignored_columns, axis=1)
        break
    except KeyError:
        print("Error : Column was not found in dataframe")

while True:  # TARGET COLUMNS
    target_output = input("Enter target output column (0 if none): ")
    if target_output.strip() != "0":
        if target_output in dataframe.columns:
            target = dataframe_copy[target_output]
            features = dataframe_copy.drop(columns=[target_output])
            break
        else:
            print("Error: Target output was not found in dataframe columns")
    else:
        target = None
        features = dataframe
        break

while True:  # MODEL SELECTION
    try:
        required_model = int(input("1. Tree,Ensemble\n"
                                   "2. Linear,SVM\n"
                                   "3. KNN,Deep\n"
                                   "Enter model number: "))
        if required_model in [1, 2, 3]:
            break
        else:
            print("Invalid model. Try again.")
    except ValueError:
        print("Invalid input. Please enter a number")

preprocessor = general_preprocessing(required_model, dataframe_copy, features)
processed_data = preprocessor.transform(features)

if hasattr(processed_data, "toarray"):
    processed_data = processed_data.toarray()

transformed_dataframe = pd.DataFrame(processed_data, columns=preprocessor.get_feature_names_out())
transformed_dataframe.to_csv("processed_data.csv", index=False)
print("Data successfully preprocessed")