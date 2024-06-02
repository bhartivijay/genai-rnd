from langchain_google_genai import GoogleGenerativeAI
from langchain import PromptTemplate, LLMChain
import streamlit as st

key = 'AIzaSyBGHrM4SXVLJlIi7Xb8bNZGX18VxEexvU8'
llm = GoogleGenerativeAI(model="models/text-bison-001",google_api_key=key, temperature=0.2)

def give_answer(question):
  template = """question: {question}
  Answer: let me think step by step"""
  prompt = PromptTemplate(template=template, input_variables=["question"])
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  response = llm_chain.run(question)
  return response

hide_github_icon ="""
<style>
header{
  visibility: hidden;
}
</style>
<style>
#manage-app-button{
visibility: hidden;
}
</style>
"""


st.markdown(hide_github_icon, unsafe_allow_html=True)
st.header("Ask me a question and I will answer")
text_input = st.text_input("Enter Question:")
value = give_answer(text_input)
st.write(value)
