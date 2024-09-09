import os
import re
import pandas as pd
from llama_index.readers.file import PandasCSVReader
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from transform_csv import transform_csv, FOLDER_NAME, FILE_NAME
from dotenv import load_dotenv

SOURCE_FILE_NAME = "data/gdp_data.csv" 
PERSIST_DIR = "./storage" 

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

file_path = os.path.join(FOLDER_NAME, FILE_NAME)

if not os.path.exists(file_path):
    transform_csv(SOURCE_FILE_NAME)
else:
    print(f"File '{file_path}' exists.")

llm = OpenAI(model="gpt-4o", temperature=0)
 
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# global settings
Settings.llm = llm
Settings.embed_model = embed_model

#Settings.chunk_size = 512
#Settings.chunk_overlap = 50

# creating chat memory buffer
memory = ChatMemoryBuffer.from_defaults(token_limit=4500)

def load_gdp_data():
    df = pd.read_csv(file_path)
    
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)

    return df

# check if storage already exists
def create_load_index():
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        parser = PandasCSVReader()
        file_extractor = {".csv": parser}
        documents = SimpleDirectoryReader(
            FOLDER_NAME, file_extractor=file_extractor
        ).load_data(show_progress=True)
        #documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    
    return index

def generate_parameter_extraction_prompt(user_query):
    prompt = f"""
    Extract the following parameters from the user query: 'GDP', 'Country Name', 'Country Code', 'Year'.

    Example 1:
    Query: "What was the GDP of the USA in 2020?"
    - Type: gdp
    - Country Name: United States
    - Country Code: USA
    - Year: 2020

    Example 2:
    Query: "Show the top 5 countries by GDP in 2019."
    - Type: top
    - Country Name: None
    - Country Code: None
    - Year: 2019
    - Top N: 5

    Example 3:
    Query: "Compare the GDP of China and India from 2000 to 2020."
    - Type: compare
    - Country Name: China, India
    - Country Code: CHN, IND
    - Start Year: 2000
    - End Year: 2020

    Example 4:
    Query: "What was the average GDP growth rate of Brazil from 2000 to 2010?"
    - Type: average_growth
    - Country Name: Brazil
    - Country Code: BRA
    - Start Year: 2000
    - End Year: 2010

    Example 5:
    Query: "What is the global GDP growth trend?"
    - Type: global_trend

    Example 7:
    Query: "What is the GDP distribution for all countries in 2018?"
    - Type: distribution
    - Year: 2018

    Now extract parameters from the following query:
    Query: "{user_query}"
    - Type:
    - Country Name:
    - Country Code:
    - Year:
    - Start Year:
    - End Year:
    - Top N:
    """
    return prompt

def extract_parameters_with_llm(engine, user_query):
    """
    Extracts parameters such as GDP, Country Name, Year, etc., from the user query using an LLM.
    """
    prompt = generate_parameter_extraction_prompt(user_query)
    llm_response = engine.chat(prompt)
    llm_text = llm_response.response.strip()
    print(llm_text)

    params = {
        'type': 'gdp',
        'country_code': None,
        'country_name': None,
        'year': None,
        'start_year': None,
        'end_year': None,
        'top_n': None
    }
    
    type_match = re.search(r"- Type: ([\w\s,]+)", llm_text)
    if type_match:
        params['type'] = type_match.group(1).lower().strip()
    
    country_name_match = re.search(r"- Country Name: ([\w\s,]+)", llm_text)
    if country_name_match:
        params['country_name'] = country_name_match.group(1).strip()

    country_code_match = re.search(r"- Country Code: ([\w\s,]+)", llm_text)
    if country_code_match:
        params['country_code'] = country_code_match.group(1).strip()

    year_match = re.search(r"- Year: (\d{4})", llm_text)
    if year_match:
        params['year'] = int(year_match.group(1))

    start_year_match = re.search(r"- Start Year: (\d{4})", llm_text)
    if start_year_match:
        params['start_year'] = int(start_year_match.group(1))

    end_year_match = re.search(r"- End Year: (\d{4})", llm_text)
    if end_year_match:
        params['end_year'] = int(end_year_match.group(1))

    top_n_match = re.search(r"- Top N: (\d+)", llm_text)
    if top_n_match:
        params['top_n'] = int(top_n_match.group(1))

    return params


def process_query(query, df, index, engine):
    """
    Processes the user query using LLM for analysis and the index for data retrieval.
    """
    try:
        params = extract_parameters_with_llm(engine, query)
        
        query_type = params.get('type', '')
        country_code = params.get('country_code', '').strip()
        year = params.get('year', None)
        top_n = params.get('top_n', None)
        start_year = params.get('start_year', None)
        end_year = params.get('end_year', None)

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        country_name = None
        if country_code:
            country_row = df[df['Country Code'].str.strip() == country_code]
            if not country_row.empty:
                country_name = country_row['Country Name'].iloc[0].strip()

        if query_type == 'gdp':
            if country_name and year:
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df = df.dropna(subset=['Year'])
                df['Year'] = df['Year'].astype(int)
                df['Country Name'] = df['Country Name'].str.strip().str.lower()
                
                result = df[(df['Country Name'] == country_name.lower()) & (df['Year'] == int(year))]
                if not result.empty:
                    response = f"GDP of {country_name} in {year} was {result.iloc[0]['GDP']}."
                else:
                    response = f"No data found for {country_name} in {year}."

        elif query_type == 'top':
            if top_n and year:
                top_countries = df[df['Year'] == int(year)].sort_values(by='GDP', ascending=False).head(int(top_n))
                print(top_countries)
                response = "\n".join([f"{row['Country Name']}: {row['GDP']}" for _, row in top_countries.iterrows()])
                

        elif query_type == 'compare':
            if country_code and start_year and end_year:
                countries = [c.strip() for c in country_code.split(',')]
                results = []
                for code in countries:
                    country_row = df[df['Country Code'].str.strip() == code]
                    if not country_row.empty:
                        country_name = country_row['Country Name'].iloc[0]
                        result = df[(df['Country Name'].str.lower() == country_name.lower()) & (df['Year'] >= int(start_year)) & (df['Year'] <= int(end_year))]
                        if not result.empty:
                            gdp_data = result[['Year', 'GDP']].to_dict('records')
                            results.append({country_name: gdp_data})
                        else:
                            results.append({country_name: "No data found"})
                response = results

        elif query_type == 'average_growth':
            if country_name and start_year and end_year:
                country_data = df[(df['Country Name'].str.lower() == country_name.lower()) & (df['Year'] >= int(start_year)) & (df['Year'] <= int(end_year))]
                if country_data.empty:
                    response = f"No data available for {country_name} between {start_year} and {end_year}."
                else:
                    growth_rates = country_data['GDP'].pct_change().dropna()
                    avg_growth_rate = growth_rates.mean() * 100
                    response = f"The average GDP growth rate for {country_name} from {start_year} to {end_year} was {avg_growth_rate:.2f}%."

        elif query_type == 'global_trend':
            global_gdp = df.groupby('Year')['GDP'].sum()
            trend = global_gdp.pct_change().dropna().mean() * 100
            response = f"The average global GDP growth rate is {trend:.2f}%."

        elif query_type == 'distribution':
            if year:
                data = df[df['Year'] == int(year)][['Country Name', 'GDP']].sort_values(by='GDP', ascending=False)
                response = data.to_string(index=False)

        else:
            # response = "I'm not sure how to handle this request. Could you please clarify?"
            response = engine.query(query)

    except Exception as e:
        response = f"Error processing query: {str(e)}"
    
    return response

def chatbot_interface(index, df, engine):
    """
    Chatbot interface to interact with the user for GDP data queries.
    """
    print("Welcome! Ask about GDP or data analysis.")
    print("To exit, type 'exit'.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = process_query(user_input, df, index, engine)
        print(f"Bot: {response}")

# 
# if __name__ == "__main__":
#  
#     gdp_data = load_gdp_data()
#     gdp_index = create_load_index()
#     #engine = index.as_query_engine()#similarity_top_k=3)
#     engine = CondensePlusContextChatEngine.from_defaults(    
#        gdp_index.as_retriever(),    
#        memory=memory,    
#        llm=llm
#     )

#     chatbot_interface(gdp_index, gdp_data, engine)