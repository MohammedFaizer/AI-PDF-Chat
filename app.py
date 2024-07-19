import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(conversation_history):
    history_text = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in conversation_history])
    prompt_template = f"""
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, just say, "answer is not available in the context". Don't provide the wrong answer.\n\n
    Context:\n{{context}}\n
    Previous Q&A:\n{history_text}\n
    Question:\n{{question}}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    docs = new_db.similarity_search(user_question)
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    chain = get_conversational_chain(st.session_state.conversation_history)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.session_state.conversation_history.append({"question": user_question, "answer": response["output_text"]})
    st.write("Reply: ", response["output_text"])
    st.write("---")
    
    for qa in st.session_state.conversation_history:
        st.write("Q: ", qa["question"])
        st.write("A: ", qa["answer"])

def main():
    st.set_page_config(page_title="Cyces.AI", page_icon="🤖")
    st.header("Chat with your own PDF")

    user_question = st.text_input("Ask a related Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("extracting..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Completed successfully👍🏼")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
