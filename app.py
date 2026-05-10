import pickle
import numpy as np
import gradio as gr

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Config
MODEL_PATH = "/home/ashim-kc/ai-ml/coursework-task3/best_binary_class_model.h5"
TOKENIZER_PATH = "/home/ashim-kc/ai-ml/coursework-task3/lstm_tokenizer.pkl"

MAX_LEN = 264  


# Load model and Tokenizer
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


# Preprocessing function
def preprocess_text(text):

    # Optional cleaning
    text = text.lower()

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return padded


# Prediction function
def predict_sentiment(review):

    if not review.strip():
        return "Please enter a review.", ""

    processed = preprocess_text(review)

    prediction = model.predict(processed, verbose=0)[0][0]

    if prediction <= 0.5:
        label = "Positive 😊"
        confidence = prediction
    else:
        label = "Negative 😞"
        confidence = 1 - prediction

    confidence_text = (
        f"Confidence: {confidence * 100:.2f}%\n"
        f"Raw score: {prediction:.4f}"
    )

    return label, confidence_text


# Gradio UI
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=6,
        placeholder="Write a hotel review here..."
    ),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Details")
    ],
    title="🏨 Hotel Review Sentiment Analyzer",
    description=(
        "BiLSTM model for hotel review "
        "binary sentiment classification."
    ),
    examples=[
        ["The room was extremely clean and the staff were very helpful."],
        ["Terrible service, dirty bathroom and noisy rooms."],
        ["The hotel location was amazing and breakfast was great."]
    ]
)


demo.launch(share = True)