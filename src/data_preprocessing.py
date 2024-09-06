import pandas as pd

def load_and_clean_data():
    # Load the main GDP dataset
    gdp_data = pd.read_csv('data/API_NY/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3403845.csv', skiprows=4)

    # Load the metadata for countries
    metadata_country = pd.read_csv('data/API_NY/Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3403845.csv')

    # Load the metadata for indicators
    metadata_indicator = pd.read_csv('data/API_NY/Metadata_Indicator_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3403845.csv')

    gdp_data = gdp_data.dropna(how='all', axis=1)
    gdp_data.fillna(0, inplace=True)

    gdp_data.columns = gdp_data.columns.str.strip()
    gdp_data = gdp_data.apply(pd.to_numeric, errors='ignore')

    # print(gdp_data.info())

    return gdp_data, metadata_country, metadata_indicator

# if __name__ == "__main__":
#     gdp_data, metadata_country, metadata_indicator = load_and_clean_data()
#     print(gdp_data.head())  
