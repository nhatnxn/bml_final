import pickle
import streamlit as st
from api import sentiment_classify
root_path = "C:/Users/khiemlh/Desktop/DraftSpace/Sentiment Analysis/"

tfidf_vectorizer = pickle.load(open(root_path + "ckps/TfidfVectorizer.pkl", "rb"))
model = pickle.load(open(root_path + "ckps/SVC.pkl", "rb"))

st.markdown("# **Phân Tích Quan Điểm Văn Bản Tiếng Việt**")

st.markdown("### **Input:**")
comment = st.text_input("Nhập một bình luận:")
gif_path = ""
if len(comment) > 0:
    pred = sentiment_classify(comment, tfidf_vectorizer, model)
    if pred == 0:
        st.markdown("### **Output: Negative**")
        gif_path = "gifs/Negative.gif"
        st.image(gif_path, width=200)
    if pred == 1:
        st.markdown("### **Output: Neutral**")
        gif_path = "gifs/Neutral.gif"
        st.image(gif_path, width=200)
    if pred == 2:
        st.markdown("### **Output: Positive**")
        gif_path = "gifs/Positive.gif"
        st.image(gif_path, width=200)