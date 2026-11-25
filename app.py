
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Replace with your actual Hugging Face username
MODEL_ID = "merahb126/movie-review-sentiment-distilbert"

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬"
)

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review to predict its sentiment.")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as exc:
    st.error("Could not load the Hugging Face model. Check MODEL_ID.")
    st.exception(exc)
    st.stop()

review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if review.strip():
        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]
            prediction = int(torch.argmax(probabilities))

        if prediction == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 🙁")

        st.write(
            f"Confidence: {probabilities[prediction].item():.2%}"
        )
    else:
        st.warning("Please enter a movie review.")
