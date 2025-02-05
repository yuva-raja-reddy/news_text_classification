import gradio as gr
import pickle
import gensim
import numpy as np

# Load the trained models
with open("rf_model.pkl", "rb") as model_file:
    rf_loaded = pickle.load(model_file)

w2v_model_loaded = gensim.models.Word2Vec.load("w2v_model.model")

# Function to preprocess, vectorize, and predict the class
def predict_news_category(text):
    """
    Takes raw text as input, preprocesses it, converts it into a vector,
    and predicts the news category using the trained Random Forest model.
    """
    # Preprocess text
    processed_text = gensim.utils.simple_preprocess(text)

    # Convert text to vector
    def sentence_vectorizer(sentence, model):
        bucket = np.zeros(model.vector_size)
        counter = 0
        for word in sentence:
            if word in model.wv:
                counter += 1
                bucket += model.wv[word]
        if counter > 0:
            bucket /= counter
        return bucket

    text_vector = sentence_vectorizer(processed_text, w2v_model_loaded).reshape(1, -1)

    # Predict class
    predicted_label = rf_loaded.predict(text_vector)[0]

    return predicted_label

# List of example news texts
examples = [
    ["The government is introducing new policies for healthcare reforms."],
    ["The stock market saw a significant rise after the tech boom."],
    ["The latest football match between Manchester United and Liverpool was thrilling."],
    ["A new movie featuring top Hollywood actors is set to release this weekend."],
    ["NASA's latest space mission has successfully landed on Mars."]
]

# Define Gradio interface
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>📰 News Category Classifier</h1>")
    gr.HTML("<p style='text-align: center;'>Enter a news article and get its category prediction.</p>")
    
    with gr.Row():
        text_input = gr.Textbox(label="Enter news text", interactive=True)

    with gr.Row():
        submit_btn = gr.Button("Predict News Category")

    output_label = gr.Label(label="Predicted Category")

    # Example inputs
    gr.Examples(examples=examples, inputs=text_input, label="Click an example to try")

    submit_btn.click(fn=predict_news_category, inputs=text_input, outputs=output_label)

# Launch app
if __name__ == "__main__":
    demo.launch()