
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gradio as gr


# -----------------------------
# 1. Load the BERT model
# -----------------------------

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


# -----------------------------
# 2. Sentiment prediction function
# -----------------------------

def predict_sentiment(review):

    if not review.strip():
        return "Please enter a review."

    # Convert text into tokens
    tokens = tokenizer.encode(
        review,
        return_tensors="pt"
    )

    # Make prediction
    with torch.no_grad():
        result = model(tokens)

    # Get the class with the highest score
    score = int(torch.argmax(result.logits)) + 1

    # Convert score into a readable label
    labels = {
        1: "Very Negative 😡",
        2: "Negative 🙁",
        3: "Neutral 😐",
        4: "Positive 🙂",
        5: "Very Positive 😍"
    }

    return f"{labels[score]} — Score: {score}/5"


# -----------------------------
# 3. Create Gradio interface
# -----------------------------

interface = gr.Interface(
    fn=predict_sentiment,

    inputs=gr.Textbox(
        lines=5,
        placeholder="Write a review here...",
        label="Your Review"
    ),

    outputs=gr.Textbox(
        label="Sentiment"
    ),

    title="🤖 BERT Sentiment Analyzer",

    description=(
        "Enter a review and BERT will predict "
        "its sentiment from 1 to 5."
    )
)


# -----------------------------
# 4. Launch the application
# -----------------------------

interface.launch()
