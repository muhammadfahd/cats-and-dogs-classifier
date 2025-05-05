# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image

# # Load the trained model
# model = load_model('cat_dog_classifier.h5')

# # Function to preprocess the uploaded image and make a prediction
# def prepare_image(img):
#     img = img.resize((150, 150))  # Resize image to 150x150 as expected by the model
#     img_array = np.array(img) / 255.0  # Normalize the image
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

# # Streamlit app interface
# st.title("Cat vs Dog Image Classifier")

# st.write("Upload an image of a cat or a dog, and the model will predict the class.")

# # Upload an image file
# uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# if uploaded_image is not None:
#     # Open the image using PIL
#     img = Image.open(uploaded_image)
    
#     # Display the uploaded image
#     st.image(img, caption="Uploaded Image", use_column_width=True)
    
#     # Prepare the image and make a prediction
#     img_array = prepare_image(img)
#     prediction = model.predict(img_array)
    
#     # Interpret the prediction
#     if prediction[0] > 0.5:
#         st.write("Prediction: **Dog**")
#         st.write(f"Confidence: {prediction[0][0]*100:.2f}%")
#     else:
#         st.write("Prediction: **Cat**")
#         st.write(f"Confidence: {(1 - prediction[0][0])*100:.2f}%")



import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('cat_dog_classifier.h5')

# Function to preprocess the uploaded image
def prepare_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Page title and description
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🐾 Cat vs Dog Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image and let the model tell you whether it's a cat or a dog!</p>", unsafe_allow_html=True)
st.markdown("---")

# File uploader
uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)

    # Layout: Image on the left, result on the right
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        img_array = prepare_image(img)
        prediction = model.predict(img_array)

        # Determine class and confidence
        if prediction[0] > 0.5:
            label = "🐶 Dog"
            confidence = prediction[0][0] * 100
        else:
            label = "🐱 Cat"
            confidence = (1 - prediction[0][0]) * 100

        # Highlight result
        st.markdown(f"<h2 style='color: #2E8B57;'>Prediction: {label}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: #555;'>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)
