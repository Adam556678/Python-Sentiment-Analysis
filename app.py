import gradio as gr
from transformers import pipeline
import pandas as pd


# =============================================================================
# 1. LOAD THE PIPELINE FROM HUGGING FACE
# =============================================================================
# This will download the model and tokenizer from the Hub and load them into
# a ready-to-use pipeline object. This happens only ONCE when the app starts.

# model link
MODEL_NAME = "Adam0x75/distilbert-finetuned-sentiment-analysis"

print(f"Loading model: {MODEL_NAME}...")
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit or handle the error appropriately
    sentiment_pipeline = None


# =============================================================================
# 2. DEFINE THE PREDICTION FUNCTION
# =============================================================================
# This function will be called by Gradio every time a user submits input.
# It uses the pre-loaded pipeline to make predictions.

def predict_sentiment(text):
    """
    Takes raw text as input and returns a dictionary of sentiment probabilities.
    """
    if not sentiment_pipeline:
        return "Error: Model is not available."
    
    # The pipeline returns a list of dictionaries. We need all scores.
    results = sentiment_pipeline(text, return_all_scores=True)
    
    
    # Define a mapping from the model's output labels to human-readable names
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    
    # Use a dictionary comprehension to create the formatted output
    formatted_results = {label_map[item['label']]: item['score'] for item in results[0]}
    
    return formatted_results


# =============================================================================
# 3. CREATE AND LAUNCH THE GRADIO INTERFACE
# =============================================================================

def main():
    """
    Main function to create and launch the Gradio app.
    """
    demo = gr.Interface(
        fn=predict_sentiment,
        inputs=gr.Textbox(
            placeholder="Enter a sentence to analyze its sentiment...", 
            label="Input Text"
        ),
        outputs=gr.Label(
            label="Predicted Sentiments"
        ),
        title="📈 Sentiment Analysis with DistilBERT",
        description="""
        This is a demo of a sentiment analysis model fine-tuned using DistilBERT. 
        It classifies text into three categories: Negative, Neutral, or Positive.
        The model was trained on a custom dataset and is hosted on the Hugging Face Hub.
        """,
        examples=[
            ["The movie was fantastic! I would recommend it to everyone."],
            ["I'm not sure how I feel about this product."],
            ["This was the worst customer service experience I have ever had."]
        ],
        allow_flagging="never"
    )

    # Launch the app!
    print("Launching Gradio app...")
    demo.launch()

if __name__ == "__main__":
    main()