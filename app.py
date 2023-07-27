from dotenv import load_dotenv
import os, streamlit as st

# Load OpenAI API key in env variable
load_dotenv()

from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI

# Define a simple Streamlit app
st.title("Ask Paul Llama")
query = st.text_input("What do you want to know?", "")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            # Define LLM  by using text-davinci-003 by default
            llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003"))

            # Configure prompt parameters and initialise helper
            max_input_size = 4096 # maximum input size
            num_output = 1024 # number of output tokens (256 by default)
            max_chunk_overlap = 20 # maximum chunk overlap
            chunk_size_limit = 600 # chunk size limit

            # define prompt helper
            prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

            # Load documents from the 'data' directory
            # (source: 'https://github.com/dnzengou/yc-startup-playbook/tree/main/ask-yc-paul-graham')
            documents = SimpleDirectoryReader('data').load_data()
            
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
            index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
            
            #index.save_to_disk('index.json')
            
            #return index
            
            #index = GPTSimpleVectorIndex.load_from_disk('index.json')
            
            response = index.query(query)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
