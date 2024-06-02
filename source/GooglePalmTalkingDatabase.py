from langchain_google_genai import GoogleGenerativeAI
from langchain import PromptTemplate, LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import streamlit as st

key = 'AIzaSyBGHrM4SXVLJlIi7Xb8bNZGX18VxEexvU8'

llm = GoogleGenerativeAI(model="models/text-bison-001",google_api_key=key, temperature=0.2)

db_user='root'
db_password="root"
db_host="localhost"
db_name="world"

db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)
db_chain = SQLDatabaseChain.from_llm(llm, db)
def give_answer(text_input):
    response = db_chain(text_input)
    return response

hide_github_icon = """
#MainMenu {
  visibility: hidden;
}
"""

st.markdown(hide_github_icon, unsafe_allow_html=True)
st.header("You can talk to you database")
text_input = st.text_input("Enter Question:")
my_answer=give_answer(text_input)
st.write(my_answer['result'])








