from langchain_google_genai import GoogleGenerativeAI
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

key = 'AIzaSyBGHrM4SXVLJlIi7Xb8bNZGX18VxEexvU8'

hide_github_icon ="""
<style>
header{
  visibility: hidden;
}

</style>
"""

conversession_chain=None
#doc=PyPDFLoader.load("asd.pdf")

def readPdfText(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore=FAISS.from_texts(text_chunks,embedding=embedding)
    return vectorstore

def getConversastonalChain(vectorstore):
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=key, temperature=0.2)
    memory=ConversationBufferMemory(memory_key="chat_history", return_message=False, return_docs=False, verbose=True, output_key='answer')
    conversastional_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory,return_source_documents=True, get_chat_history=lambda h: h)
    return conversastional_chain


def handleUserInput(user_question):
    convers_chain = st.session_state.conversation
    response=convers_chain.invoke(user_question)
    st.write(response['answer'])


st.set_page_config("Chat With Multiple PDFs", page_icon=":books:")
st.header("SwiftHR :books:")
st.markdown('Developed by Vijay Bharti')
st.markdown(hide_github_icon,unsafe_allow_html=True)
user_input = st.text_input("Ask a question from your document")

with st.sidebar:
    st.header("Upload Your PDF Document/s")

    pdfs = st.file_uploader("Upload pdf files and click process", accept_multiple_files=True, type="pdf")
    if st.button("Process"):
        with st.spinner("Process"):
            raw_text = readPdfText(pdfs)
            chunks = get_text_chunks(raw_text)
            vectorstore = get_vector_store(chunks)
            st.session_state.conversation = getConversastonalChain(vectorstore)
            st.success("Done")
if st.button("Get Answer"):

    if "conversation" not in st.session_state:
        st.warning('Please upload PDF/s to start asking questions')
    else:
        handleUserInput(user_input)

