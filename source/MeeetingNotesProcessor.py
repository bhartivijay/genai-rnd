import spacy
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import subprocess
import streamlit as st

st.header("Meeting Notes Processor")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_sm")

@st.cache_resource
def download_en_core_web_sm():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
def categorize_insight(sentence):
    """
    Categorize a sentence as actionable or non-actionable.
    Actionable sentences often contains verbs implying tasks (e.g., "follow up", "create", "send").
    """

    actionable_keywords = ["follow up","send","create","schedule","update","review"]
    non_actionable_keywords = ["discussed","highlighted","notes","mentioned"]

    for keywords in actionable_keywords:
        if keywords in sentence.lower():
            return "Actionable"

    for keywords in non_actionable_keywords:
        if keywords in sentence.lower():
            return "Non-Actionable"

    return "Uncategorized"

def extract_entities(doc):
    """Extract named entities like clients, organizations, dates, and products."""
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return entities

def derive_insights(meeting_notes):
    """Process meeting notes to derive actionable anf non-actionable insights."""
    sentences = sent_tokenize(meeting_notes)
    insights = {
        "Actionable" : [],
        "Non-Actionable": [],
        "Uncategorized" : []
    }

    entities = []

    for sentence in sentences:
        category = categorize_insight(sentence)
        insights[category].append(sentence)

        doc = nlp(sentence)
        entities.append(extract_entities(doc))

    return insights, entities

metting_notes = """
Client ABC emphasized the importance of delivering the project by next month.
Follow up with design team to finalize the UI mockups. 
"""

insights, entities = derive_insights(metting_notes)

st.write("\n----Actionable Insights ----")
for insight in insights["Actionable"]:
    st.write(f"- {insight}")

st.write("\n----Non-Actionable Insights ----")
for insight in insights["Non-Actionable"]:
    st.write(f"- {insight}")

st.write("\n----Uncategorized Insights ----")
for insight in insights["Uncategorized"]:
    st.write(f"- {insight}")

st.write("\n----Extracted Entities ----")
for entity_group in entities:
    for label, text in entity_group.items():
        st.write(f"{label}: {', '.join(text)}")
