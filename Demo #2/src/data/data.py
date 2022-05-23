import pandas as pd


# Columns
original_columns = [
    "User_ID",
    "Product_ID",
    "Gender",
    "Age",
    "Occupation",
    "City_Category",
    "Stay_In_Current_City_Years",
    "Marital_Status",
    "Product_Category_1",
    "Product_Category_2",
    "Product_Category_3",
    "Purchase",
]

features = [
    "User_ID",
    "Product_ID",
    "Gender",
    "Age",
    "Occupation",
    "City_Category",
    "Stay_In_Current_City_Years",
    "Marital_Status",
    "Product_Category_1",
    "Product_Category_2",
    "Product_Category_3",
]

target_label = "Purchase"

categorical_features = [
    "User_ID",
    "Product_ID",
    "Gender",
    "Age",
    "Occupation",
    "City_Category",
    "Marital_Status",
    "Product_Category_1",
    "Product_Category_2",
    "Product_Category_3",
]

numeric_features = ["Stay_In_Current_City_Years"]


def import_data(
    file=None, bucket_name="", directory=""
):
    """Import train or test data from Cloud Storage

    :param file: filename
    :param bucket_name: bucket name
    :param directory: path of the file folder in the bucket
    :return: Pandas dataframes of the features & label
    """
    # data = pd.read_csv("gs://%s/%s/%s.csv" % (bucket_name, directory, file))
    #TODO: Load data from GCP
    train_data = pd.read_csv('D:/Data/BlackFriday/train.csv')
    X, y = train_data[features], train_data[target_label]
    return X, y


def clean(df):
    """Clean the values of Stay_In_Current_City_Years variable into a numeric type
    """
    values_mapping = {"0": 0, "1": 1, "2": 2, "3": 3, "4+": 4}
    df["Stay_In_Current_City_Years"] = df["Stay_In_Current_City_Years"].map(
        values_mapping
    )
    return df


def cast(df):
    """Cast the categorical features into strings
    """
    df['Product_Category_1'] = df['Product_Category_1'].astype(str)
    df['Product_Category_2'] = df['Product_Category_2'].astype(str, errors='ignore')
    df['Product_Category_3'] = df['Product_Category_3'].astype(str, errors='ignore')
    return df
    # return df.astype(str, errors="ignore")
    