from gdp_chat import (load_gdp_data, 
                      create_load_index, 
                      CondensePlusContextChatEngine,
                      llm, 
                      memory, 
                      chatbot_interface)

if __name__ == "__main__":
    # Load GDP data and create or load the index
    gdp_data = load_gdp_data()
    gdp_index = create_load_index()
    
    # Initialize the chat engine with memory and LLM
    engine = CondensePlusContextChatEngine.from_defaults(    
       gdp_index.as_retriever(),    
       memory=memory,    
       llm=llm
    )
    # Start the chatbot interface
    chatbot_interface(gdp_index, gdp_data, engine)
