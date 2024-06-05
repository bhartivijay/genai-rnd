from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import langchain
import streamlit as st

hide_github_icon ="""
<style>
header{
  visibility: hidden;
}

</style>
"""


key = 'AIzaSyBGHrM4SXVLJlIi7Xb8bNZGX18VxEexvU8'

llm = GoogleGenerativeAI(model="models/text-bison-001",google_api_key=key, temperature=0.2)

def process_url_and_provide_response(website_url_list,question):
    loader=UnstructuredURLLoader(urls=website_url_list)
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=10)
    docs=text_splitter.split_documents(data)
    embeddings=HuggingFaceEmbeddings()
    vector_str=FAISS.from_documents(docs,embeddings)
    #chain=RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_str.as_retriever())
    #retriever=vector_str.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1})
    retriever=vector_str.as_retriever(search_kwargs={"k": 5})
    langchain.debug=True
    query=question
    docs=retriever.invoke(query)
    final_response=''
    for doc in docs:
        final_response=final_response+'\n'+doc.page_content
    return final_response

#st.markdown(hide_github_icon, unsafe_allow_html=True)
st.header("Ask question from provided website")
st.markdown('Developed by Vijay Bharti')
website_url = st.text_input("Enter Website URL:")
website_url_list=[]
if ',' in website_url:
    website_url_list=website_url.split()
else:
    website_url_list.append(website_url)

question = st.text_input("Enter Question:")
if(len(website_url_list)>0 and question!=''):
    final_response=process_url_and_provide_response(website_url_list,question)
    st.write(final_response)
