import streamlit as st
from annotated_text import annotated_text
import requests

def parser_predict(res):
    out_res = []
    for i in range(0, len(res)):
        if res[i]["label"] != "O":
            out_res.append((res[i]["word"], res[i]["label"]))
        else:
            out_res.append(" " + res[i]["word"])

    return out_res

st.set_page_config(layout="wide")

"""
# NER (DeepPavlov/ruBERT + surdan/nerel_short) Demo:
"""

text = st.text_input("Text:", value="")
result = requests.post('http://backend:8090/predict', params={'text': text}).json()
pred = parser_predict(result)

"""
Annotated text:
"""

annotated_text(
    *pred
)
