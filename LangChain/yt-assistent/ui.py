# streamlit_app.py
import streamlit as st
from main import process_video_query

st.title("YouTube Video Query Answering Application")

video_url = st.text_input("YouTube Video URL", "")
question = st.text_input("Question", "")

if st.button("Get Answer"):
    if video_url and question:
        response = process_video_query(video_url, question)
        st.write(response)
    else:
        st.error("Please provide both a YouTube Video URL and a question.")
