import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from PIL import Image

st.set_page_config(page_title="Signature Authentication")

st.title("✍️ Personal Signature Authentication System")

st.write("Step 1: Upload your genuine signature samples")
st.write("Step 2: Upload a signature to verify")

IMG_SIZE = (200, 100)

# --------------------------
# Feature Extraction
# --------------------------

def extract_features(image):

    image = cv2.resize(image, IMG_SIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2)
    )

    return features


# --------------------------
# Upload Genuine Signatures
# --------------------------

genuine_files = st.file_uploader(
    "Upload Genuine Signatures (Minimum 5)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

model = None

if genuine_files and len(genuine_files) >= 5:

    X = []
    Y = []

    for file in genuine_files:
        image = Image.open(file)
        img = np.array(image)

        features = extract_features(img)
        X.append(features)
        Y.append(1)  # Genuine label

    # Add fake class dummy (to allow training)
    # We'll generate slight noise to create negative class
    for file in genuine_files[:3]:
        image = Image.open(file)
        img = np.array(image)
        img = cv2.GaussianBlur(img, (9,9), 0)

        features = extract_features(img)
        X.append(features)
        Y.append(0)  # Fake class

    X = np.array(X)
    Y = np.array(Y)

    model = SVC(kernel="rbf", probability=True)
    model.fit(X, Y)

    st.success("Model trained successfully using your genuine signatures ✅")


# --------------------------
# Upload Test Signature
# --------------------------

if model:

    test_file = st.file_uploader(
        "Upload Signature to Verify",
        type=["jpg", "png", "jpeg"]
    )

    if test_file:

        image = Image.open(test_file)
        img = np.array(image)

        st.image(image, caption="Uploaded Signature", width=300)

        features = extract_features(img)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()

        if prediction == 1:
            st.success(f"✅ VALID Signature ({probability*100:.2f}%)")
        else:
            st.error(f"❌ FORGED Signature ({probability*100:.2f}%)")
