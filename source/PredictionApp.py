import streamlit as st
import os
import time
from langchain import HuggingFaceHub, PromptTemplate, LLMChain


def answerTheQuestion(question):
  template = """question: {question}
  Answer: let me think step by step"""
  prompt = PromptTemplate(template=template, input_variables=["question"])
  llm = HuggingFaceHub(repo_id="google/flan-t5-base",
                       model_kwargs={
                           "temperature": 0.9,
                           "max_length": 64
                       })
  llm_chain = LLMChain(prompt=prompt, llm=llm)

  response = llm_chain.run(question)
  return response

st.header("My prediction app")
text_input = st.text_input("Enter some text")
value=answerTheQuestion(text_input)
st.write(value)