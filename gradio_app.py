import argparse

import gradio as gr

from inference import load_model, predict_news

model, vocab, max_length = load_model()


def classify(text):
    result = predict_news(text or "", model, vocab, max_length)
    if result is None:
        raise gr.Error("Please enter a news article with some valid words.")
    label = {
        "Fake News": result["fake_probability"],
        "Real News": result["real_probability"],
    }
    return label


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=10, label="News text", placeholder="Paste or type a news article here..."),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="📰 Fake News Detection (BiLSTM)",
    description=(
        "Classifies news articles by language patterns. It is not a fact-checker; "
        "the probabilities are model scores, not proof that a claim is true or false."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link (use in Colab)")
    args = parser.parse_args()
    demo.launch(share=args.share)
