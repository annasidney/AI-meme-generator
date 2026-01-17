import streamlit as st
from PIL import Image
import base64
import io
import openai
import cv2
import numpy as np

# OpenAI API key is loaded securely using environment variables


def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_caption(image_file):
    base64_image = encode_image(image_file)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "See this image and make a simple understandable 6 worded caption of what is happening there"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                },
            ]
        }]
    )

    return response.choices[0].message["content"].strip()

def generate_humorous_caption(base_caption):
    prompt = f"Make this caption funny and meme-worthy: {base_caption}"

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "based on that base text create a simple short 6 worded funny meme caption"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response["choices"][0]["message"]["content"].strip()

st.title("AI Meme Maker: Turn Images into Laughter!")



uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    if "image" not in st.session_state:
        image = Image.open(uploaded_image)
        st.session_state["image"] = image
        st.session_state["base_caption"] = generate_image_caption(io.BytesIO(uploaded_image.getvalue()))
        st.session_state["humorous_caption"] = generate_humorous_caption(st.session_state["base_caption"])

    st.image(st.session_state["image"], caption="Uploaded Image", use_column_width=True)

    base_caption = st.session_state["base_caption"]
    humorous_caption = st.session_state["humorous_caption"]
    st.write("Generated Caption:", base_caption)
    st.write("Humorous Caption:", humorous_caption)

    selected_caption = st.radio("Select a Caption:", [base_caption, humorous_caption])

    st.subheader("Customize Your Caption")
    custom_caption = st.text_input("Edit your caption:", selected_caption)
    font_size = st.slider("Font Size:", 10, 50, 20)
    caption_position = st.selectbox("Caption Position:", ["Top", "Bottom", "Center"])

    st.session_state["custom_caption"] = custom_caption
    st.session_state["font_size"] = font_size
    st.session_state["caption_position"] = caption_position

    if st.button("Update Meme"):
        image_cv = np.array(st.session_state["image"])
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)  
        thickness = 3

        text = st.session_state["custom_caption"]
        text_size = cv2.getTextSize(text, font, 1, thickness)[0]
        text_width, text_height = text_size

        if st.session_state["caption_position"] == "Top":
            position = (int((image_cv.shape[1] - text_width) / 2), text_height + 10)
        elif st.session_state["caption_position"] == "Bottom":
            position = (int((image_cv.shape[1] - text_width) / 2), image_cv.shape[0] - 10)
        else:  
            position = (int((image_cv.shape[1] - text_width) / 2), int((image_cv.shape[0] + text_height) / 2))

        cv2.putText(image_cv, text, position, font, st.session_state["font_size"] / 20, font_color, thickness, lineType=cv2.LINE_AA)

        image_with_text = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        st.image(image_with_text, caption="Meme Preview", use_column_width=True)

        byte_im = io.BytesIO()
        image_with_text.save(byte_im, format='PNG')
        byte_im.seek(0)

        st.download_button("Download Meme", byte_im, file_name="meme.png", mime="image/png")

