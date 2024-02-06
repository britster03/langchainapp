import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from supabase import create_client, Client
import torch

supabase_url = 'https://rquwntqrmfmwtzzlbjci.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJxdXdudHFybWZtd3R6emxiamNpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDExMDYzNTEsImV4cCI6MjAxNjY4MjM1MX0.szFlkP1hTlddGoE8akJrt78fCjB1XVIhWF8ZrKCoxZw'

supabase: Client = create_client(supabase_url, supabase_key)

load_dotenv()
def main():
    st.header("Chat with PDF 💬")
    
    # Upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
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
        st.write(store_name)
        embeddings = HuggingFaceEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Continuous query input area
        query_counter = 0
        
        while True:
            query_counter += 1
            query_key = f"query_text_area_{query_counter}"
            query = st.text_area("Ask a question about your PDF file:", key=query_key)
            
            if not query:
                st.warning("Please ask a question.")
                break  # Exit the loop if the user does not provide a query
            
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

                thumbs_up_key = f"thumbs_up_button_{query_counter}"
                thumbs_down_key = f"thumbs_down_button_{query_counter}"

                thumbs_up = fed_thumbs_up.button("👍", key=thumbs_up_key)
                thumbs_down = fed_thumbs_down.button("👎", key=thumbs_down_key)

                if thumbs_up:
                    st.success("Thanks for your feedback!")
                elif thumbs_down:
                    st.info("Generating an alternative answer...")

                    # Run the QA chain again with different parameters or models
                    docs = VectorStore.similarity_search(query=query)
                    llm_alt = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.3, "max_length": 10000})
                    chain_alt = load_qa_chain(llm=llm_alt, chain_type="stuff")
                    response_alt = chain_alt.run(input_documents=docs, question=query)
                    st.write(response_alt)
                    answer_alt_start = response_alt.find("Alternative Answer:")

                    if answer_alt_start != -1:
                        answer_alt = response_alt[answer_alt_start + len("Alternative Answer:"):].strip()
                        st.write(answer_alt)
                    else:
                        st.write("No alternative answer found.")

            else:
                st.write("No answer found.")

if __name__ == '__main__':
    main()
