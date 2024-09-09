# GDP Chatbot 

This project is an AI-driven chatbot that allows users to interact with a World Bank GDP dataset. The chatbot uses a language model to interpret user queries and provides responses based on country GDP data from various years.

## Setup Instructions

### Prerequisites

Ensure you have the following tools installed:
- Python 3.8+
- [OpenAI API key](https://beta.openai.com/signup/)
- Git
- Conda (or Python virtualenv for environment management)

### 1. Create and Activate Environment

Use the provided `environment.yml` file (or manually install dependencies):

```bash
conda create -f environment.yml
conda activate gdp-chatbot
```

Alternatively, install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

Add your OpenAI API key to a `.env` file at the root of the project:

```
OPENAI_API_KEY=your_openai_api_key
```

### 3. Preprocess the GDP Data

Before running the chatbot, transform the provided `gdp_data.csv` file into a format suitable for querying. This can be done using the `transform_csv.py` script:

```bash
python src/transform_csv.py
```

The transformed CSV will be saved in the `csv_source/` folder.

### 4. Run the Chatbot

Once the data is processed, you can start the chatbot by running the `main.py` script:

```bash
python src/main.py
```

You can now ask questions about the GDP of various countries, such as:

```
> What was the GDP of the USA in 2020?
> Compare the GDP of China and India from 2000 to 2020.
> Show the top 5 countries by GDP in 2019.
```

### 5. Chatbot Commands

The chatbot can understand queries about:
- GDP for a specific country and year
- Top N countries by GDP
- GDP comparisons between countries across multiple years
- Average GDP growth rates
- Global GDP trends and distribution

### 6. Example Queries

Here are some example queries you can ask the chatbot:
- "What was the GDP of Brazil in 2010?"
- "Show the top 10 countries by GDP in 2019."
- "Compare the GDP of the USA and China from 2005 to 2015."
- "What is the global GDP growth trend?"

## File Descriptions

### `src/transform_csv.py`
This script is responsible for transforming the original `gdp_data.csv` from a wide format (with years as columns) into a long format, making it easier to work with for querying GDP values over time.

### `src/gdp_chat.py`
This file contains the main logic for interpreting user queries and fetching the relevant GDP data. It also defines how to interact with the preprocessed GDP dataset and the vector index.

### `src/main.py`
This is the entry point for running the chatbot. It sets up the environment, loads the data, and starts the chatbot interface.

## Notes

- The transformed GDP data will be saved in the `csv_source/` directory.
- Make sure to clean your Git history if you accidentally commit sensitive information (e.g., API keys).

