import streamlit as st

from inference import load_model as _load_model, predict_news


@st.cache_resource
def load_model():
    return _load_model()


# ==========================================
# 6. Streamlit user interface
# ==========================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")

st.write(
    "Classify news articles using a BiLSTM deep learning model."
)

st.info(
    "This application analyzes language patterns in news articles. "
    "It is not a complete fact-checking system and cannot guarantee "
    "whether a claim is true or false."
)

try:
    model, vocab, max_length = load_model()

except Exception as error:
    st.error(f"Failed to load the model: {error}")
    st.stop()

st.success("BiLSTM model loaded successfully.")

news_text = st.text_area(
    "Enter News Text",
    height=220,
    placeholder="Paste or type a news article here..."
)

if st.button(
    "🔍 Predict News",
    type="primary",
    use_container_width=True
):
    if not news_text.strip():
        st.warning("Please enter a news article first.")

    else:
        with st.spinner("Analyzing the news article..."):
            result = predict_news(
                news_text,
                model,
                vocab,
                max_length
            )

        if result is None:
            st.warning(
                "No valid words were found in the provided text."
            )

        else:
            st.divider()
            st.subheader("Prediction Result")

            if result["prediction"] == "Fake News":
                st.error("🚨 Predicted Class: Fake News")
            else:
                st.success("📰 Predicted Class: Real News")

            fake_pct = result["fake_probability"] * 100
            real_pct = result["real_probability"] * 100

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fake Probability",
                    f"{fake_pct:.2f}%"
                )
                st.progress(result["fake_probability"])

            with col2:
                st.metric(
                    "Real Probability",
                    f"{real_pct:.2f}%"
                )
                st.progress(result["real_probability"])

            st.caption(
                "Disclaimer: These probabilities are model scores, "
                "not proof that the news is true or false."
            )

st.divider()

st.caption(
    "Project: Fake News Detection | Model: BiLSTM | Framework: PyTorch"
)
