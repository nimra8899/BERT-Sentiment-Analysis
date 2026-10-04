# 🤖 BERT Sentiment Analysis

A beginner-friendly NLP project that uses a pretrained BERT-based sentiment classification model from Hugging Face Transformers to analyze text and predict sentiment from **1 to 5 stars**.

The project also includes a **Gradio web interface** where users can enter a sentence or review and receive a sentiment prediction.

## 🚀 Features

* Pretrained BERT sentiment model
* Hugging Face Transformers
* Text tokenization
* Sentiment classification
* 1–5 sentiment scoring
* Interactive Gradio interface
* PyTorch-based inference

## 🧠 Model

This project uses:

`nlptown/bert-base-multilingual-uncased-sentiment`

from Hugging Face.

The model is based on BERT and has been fine-tuned for multilingual sentiment classification.

### Sentiment Classes

| Score | Sentiment        |
| ----- | ---------------- |
| ⭐ 1   | Very Negative 😡 |
| ⭐ 2   | Negative 🙁      |
| ⭐ 3   | Neutral 😐       |
| ⭐ 4   | Positive 🙂      |
| ⭐ 5   | Very Positive 😍 |

## 🔄 How It Works

```text
User enters text
       ↓
BERT Tokenizer
       ↓
Token IDs
       ↓
Pretrained BERT Model
       ↓
Logits
       ↓
Argmax
       ↓
Sentiment Score (1–5)
       ↓
Gradio Interface
```

## 🛠️ Technologies

* Python
* PyTorch
* Hugging Face Transformers
* BERT
* Gradio
* Google Colab / Jupyter Notebook

## 📁 Project Structure

```text
BERT-Sentiment-Analysis/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── sentiment_analysis.ipynb
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/nimra8899/BERT-Sentiment-Analysis.git
```

Move into the project directory:

```bash
cd BERT-Sentiment-Analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Run:

```bash
python app.py
```

Gradio will provide a local URL. Open the URL in your browser to use the sentiment analyzer.

## 🧪 Example

### Input

```text
The food was amazing and the service was excellent!
```

### Output

```text
Very Positive 😍 — Score: 5/5
```

Another example:

```text
The food was terrible and the service was very disappointing.
```

Output:

```text
Very Negative 😡 — Score: 1/5
```

## 🔍 Prediction Process

The input text is first converted into token IDs using the BERT tokenizer:

```python
tokens = tokenizer.encode(
    review,
    return_tensors="pt"
)
```

The token IDs are then passed to the pretrained model:

```python
result = model(tokens)
```

The model produces logits for the five sentiment classes.

The class with the highest logit is selected using:

```python
torch.argmax(result.logits)
```

The result is then converted into a 1–5 sentiment score.

## 📚 Learning Outcomes

This project helped me understand:

* NLP basics
* Tokens and tokenization
* Token IDs
* BERT
* Pretrained models
* Hugging Face Transformers
* Sequence classification
* Logits
* `argmax`
* Sentiment analysis
* PyTorch inference
* Building an NLP interface with Gradio

## ⚠️ Limitations

* The model may struggle with sarcasm or complex context.
* Predictions are not always perfectly accurate.
* Very long inputs may require additional processing.
* The model's output should be treated as a prediction rather than absolute truth.

## 🔮 Future Improvements

* Add confidence/probability scores
* Support batch sentiment analysis
* Add sentiment charts
* Allow CSV upload
* Compare BERT with DistilBERT
* Fine-tune BERT on a custom dataset
* Improve the Gradio UI
* Deploy the application online



