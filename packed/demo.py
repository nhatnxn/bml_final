import pickle
import streamlit as st
from api import sentiment_classify

tfidf_vectorizer = pickle.load(open("Vectorizer.pkl", "rb"))
model = pickle.load(open("SVC_clf.pkl", "rb"))

# st.markdown("# **Phân Tích Quan Điểm Văn Bản Tiếng Việt**")

st.markdown("### **Input:**")
comment = st.text_input("Nhập một bình luận:")
gif_path = ""
print('oks')
if len(comment) > 0:

    pred = sentiment_classify(comment, tfidf_vectorizer, model)
    print(len(comment))
    print(pred)
    if pred == 'NEG':
        st.markdown("### **Output: Negative**")
        gif_path = "gifs/Negative.gif"
        st.image(gif_path, width=200)
    if pred == 'NEU':
        st.markdown("### **Output: Neutral**")
        gif_path = "gifs/Neutral.gif"

        st.image(gif_path, width=200)
    if pred == 'POS':
        st.markdown("### **Output: Positive**")
        gif_path = "gifs/Positive.gif"

        st.image(gif_path, width=200)