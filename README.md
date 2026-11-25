
# Movie Review Sentiment Analyzer

## Project Description
This project predicts whether a movie review is positive or negative using a fine-tuned DistilBERT model.

## Dataset
IMDb Movie Reviews Dataset from Hugging Face.

Dataset: stanfordnlp/imdb

## Pretrained Model
distilbert/distilbert-base-uncased

## Fine-Tuned Model
My fine-tuned model is uploaded to Hugging Face.

## Technologies Used
- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- DistilBERT

## Application Features
- Enter a movie review.
- Analyze its sentiment.
- Display positive or negative prediction.
- Show prediction confidence.

## How to Run

Install the requirements:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Labels
0 = Negative
1 = Positive

