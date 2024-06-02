from langchain_community.llms import GooglePalm
from langchain import PromptTemplate, LLMChain
import streamlit as st

key = 'AIzaSyBGHrM4SXVLJlIi7Xb8bNZGX18VxEexvU8'

llm = GooglePalm(google_api_key=key, temperature=0.2)
def give_answer(question):
  template = """question: {question}
  Answer: let me think step by step"""
  prompt = PromptTemplate(template=template, input_variables=["question"])
  llm_chain = LLMChain(prompt=prompt, llm=llm)
  response = llm_chain.run(question)
  return response

st.header("Ask me question i will try to answer")
text_input = st.text_input("Enter Question:")
value=give_answer(text_input)
st.write(value)
