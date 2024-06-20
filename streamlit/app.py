
import streamlit as st
from transformers import pipeline

# Khởi tạo các pipeline cho phân tích cảm xúc
sentiment_classifier_vi = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_classifier_en = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

st.title("Ứng dụng Phân tích cảm xúc bình luận")

st.sidebar.title("Chọn ngôn ngữ")
language = st.sidebar.selectbox("Chọn ngôn ngữ của bình luận", ["Tiếng Việt", "Tiếng Anh"])

if language == "Tiếng Việt":
    st.header("Phân tích cảm xúc Tiếng Việt")
    user_input = st.text_area("Nhập bình luận Tiếng Việt của bạn tại đây")
    if st.button("Phân tích"):
        if user_input:
            result = sentiment_classifier_vi(user_input)
            score = result[0]['score']

            # Đưa ra kết quả tích cực hoặc tiêu cực dựa trên điểm số
            if score >= 0.5:
                sentiment = "Tích cực"
            else:
                sentiment = "Tiêu cực"

            st.write(f"Kết quả phân tích cảm xúc: {sentiment}")
            st.write(f"Điểm số: {score:.2f}")
        else:
            st.write("Vui lòng nhập bình luận để phân tích.")

elif language == "Tiếng Anh":
    st.header("Phân tích cảm xúc Tiếng Anh")
    user_input = st.text_area("Nhập bình luận Tiếng Anh của bạn tại đây")
    if st.button("Phân tích"):
        if user_input:
            result = sentiment_classifier_en(user_input)
            label = result[0]['label']
            score = result[0]['score']

            # Đưa ra kết quả tích cực hoặc tiêu cực dựa trên điểm số
            if label == "POSITIVE":
                sentiment = "Tích cực"
            else:
                sentiment = "Tiêu cực"

            st.write(f"Kết quả phân tích cảm xúc: {sentiment}")
            st.write(f"Điểm số: {score:.2f}")
        else:
            st.write("Vui lòng nhập bình luận để phân tích.")


