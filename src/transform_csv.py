import pandas as pd
import os

FOLDER_NAME = 'csv_source'
FILE_NAME = 'gdp.csv'

def transform_csv(input_file):
    """
    Function to transform a CSV file from wide format to long format,
    keeping the data for countries and years in a new file.

    Parameters:
    input_file (str): path to the source CSV file.
    """

    df = pd.read_csv(input_file)

    # Removing the "Indicator Name" and "Indicator Code" columns
    df = df.drop(columns=["Indicator Name", "Indicator Code"])

    # Transforming the data into long format (modernization)
    df_melted = pd.melt(df, 
                        id_vars=["Country Name", "Country Code"], 
                        var_name="Year", 
                        value_name="GDP")

    file_path = os.path.join(FOLDER_NAME, FILE_NAME)

    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)

    df_melted.to_csv(file_path, index=False)

    print(f"Transformation completed and the file has been saved at '{file_path}'.")