#streamlit for creating gui in python
# multiple queries
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from supabase import create_client, Client
import hashlib
import time
import os

supabase_url = 'https://rquwntqrmfmwtzzlbjci.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJxdXdudHFybWZtd3R6emxiamNpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDExMDYzNTEsImV4cCI6MjAxNjY4MjM1MX0.szFlkP1hTlddGoE8akJrt78fCjB1XVIhWF8ZrKCoxZw'

supabase: Client = create_client(supabase_url, supabase_key)

with st.sidebar:
    st.title('😋💬LLM Chat App')
    st.markdown('''
                ## About 
                This app is an LLM-powered chatbot built using:
                - [Straemlit](https://streamlit.io/)
                - [LangChain](https://python.langchain.com/)
                - [OpenAI](https://platform.openai.com/docs/models) LLM model
                ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Ronit Virwani](https://linkedin.com/in/ronitvirwani)')

load_dotenv()

def main():
    st.header("Chat with PDF 💬")
    
    # Upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    # ...

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        embeddings = HuggingFaceEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Continuous query input area
        query_counter = 0
        query_key = f"query_text_area_{query_counter}"
        query = st.text_area("Ask another question about your PDF file:", key=query_key)
        
        while query:
            query_counter += 1
            query_key = f"query_text_area_{query_counter}"

            docs = VectorStore.similarity_search(query=query)
            llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature": 0.2, "max_length": 10000})
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            answer_start = response.find("Helpful Answer:")
            
            if answer_start != -1:
                answer = response[answer_start + len("Helpful Answer:"):].strip()
                st.write(answer)

                # Feedback UI section
                st.subheader("Feedback:")
                st.write("Was this answer helpful?")
                fed_thumbs_up, fed_thumbs_down = st.columns(2)
                current_time = str(time.time())

# Feedback buttons
                thumbs_up_key = hashlib.md5(("thumbs_up_button" + current_time).encode()).hexdigest()
                thumbs_down_key = hashlib.md5(("thumbs_down_button" + current_time).encode()).hexdigest()

# Add feedback buttons with unique keys
                thumbs_up = fed_thumbs_up.button("👍", key=thumbs_up_key)
                thumbs_down = fed_thumbs_down.button("👎", key=thumbs_down_key)

# Handle feedback
                if thumbs_up or thumbs_down:
    # Determine the feedback value
                    feedback = 1 if thumbs_up else -1

    # Save feedback to Supabase database
                    feedback_data = {"question": query, "answer": answer, "feedback": feedback}
                    feedback_table = supabase.table("feedback").insert([feedback_data]).execute()
                    st.write(feedback_table)
                    st.success("Feedback submitted successfully!")


            else:
                st.write("No answer found.")
            
            # Prompt for the next query
            query = st.text_area("Ask another question about your PDF file:", key=query_key)

if __name__ == '__main__':
    main()