
import streamlit as st
import os
import tempfile
import time
from PIL import Image

# Import trained CNN prediction function
from src.predict import predict_captcha


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI CAPTCHA Recognition",
    page_icon="🔐",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        color: #2563EB;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
    }

    .result-box {
        background-color: #ecfdf5;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #10b981;
    }

    .prediction {
        font-size: 38px;
        font-weight: bold;
        color: #047857;
        letter-spacing: 8px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔐 AI CAPTCHA Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning Based CAPTCHA Text Recognition using CNN</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("About Project")

    st.write(
        "This application uses a trained Convolutional "
        "Neural Network (CNN) to recognize text from CAPTCHA images."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Architecture:** CNN")
    st.write("**Input:** 180 × 60 Grayscale")
    st.write("**Output:** 5 Characters")
    st.write("**Classes:** 36")
    st.write("**Framework:** TensorFlow / Keras")

    st.divider()

    st.caption("AI CAPTCHA Recognition | INN Project")


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Upload CAPTCHA Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a CAPTCHA image",
    type=["png", "jpg", "jpeg"],
    help="Upload a CAPTCHA image for recognition."
)


# --------------------------------------------------
# IMAGE PREVIEW
# --------------------------------------------------

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.subheader("Original CAPTCHA")

        st.image(
            image,
            caption="Uploaded CAPTCHA Image",
            width="stretch"
        )

        st.caption(
            f"Image dimensions: {image.width} × {image.height}"
        )

    with col2:
        st.subheader("Recognition")

        st.write(
            "Click the button below to recognize the CAPTCHA."
        )

        if st.button(
            "🔍 Recognize CAPTCHA",
            type="primary",
            use_container_width=True
        ):

            temp_path = None

            try:
                with st.spinner("CNN model is processing the image..."):

                    start_time = time.perf_counter()

                    # Save uploaded image temporarily
                    suffix = os.path.splitext(uploaded_file.name)[1]

                    with tempfile.NamedTemporaryFile(
                        suffix=suffix,
                        delete=False
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    # Call trained CNN model
                    prediction, confidences = predict_captcha(temp_path)

                    processing_time = time.perf_counter() - start_time

                # Display prediction
                st.success("CAPTCHA recognized successfully!")

                st.markdown(
                    f"""
                    <div class="result-box">
                        <p>Recognized CAPTCHA</p>
                        <div class="prediction">{prediction}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("")

                # Processing time
                st.metric(
                    "Processing Time",
                    f"{processing_time:.4f} seconds"
                )

                # Character-wise confidence
                st.subheader("Character-wise Confidence")

                for index, confidence in enumerate(confidences):

                    character = prediction[index]

                    st.write(
                        f"Character {index + 1}: **{character}**"
                    )

                    st.progress(float(confidence))

                    st.caption(
                        f"Confidence: {float(confidence) * 100:.2f}%"
                    )

            except Exception as e:

                st.error(f"Recognition failed: {str(e)}")

            finally:

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

else:

    st.info("Please upload a CAPTCHA image to get started.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "<center>Developed using Python, Streamlit and CNN</center>",
    unsafe_allow_html=True
)
