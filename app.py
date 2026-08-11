import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "Models/plant_disease_final.pth"


# =========================================================
# CLASS NAMES
# =========================================================

classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(classes)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(image):

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    top_probs, top_indices = torch.topk(
        probabilities,
        3
    )

    results = []

    for i in range(3):

        index = top_indices[0][i].item()

        class_name = classes[index]

        confidence = (
            top_probs[0][i].item() * 100
        )

        results.append(
            (class_name, confidence)
        )

    return results


# =========================================================
# CLEAN CLASS NAME
# =========================================================

def clean_name(name):

    return (
        name
        .replace("___", " — ")
        .replace("_", " ")
    )


# =========================================================
# LIGHT CSS ONLY
# No HTML components / divs
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #F6F8F6;
    }

    [data-testid="stSidebar"] {
        background-color: #EDF5EF;
    }

    h1, h2, h3 {
        color: #173B2C !important;
    }

    .stMarkdown p {
        color: #42584C;
    }

    .stButton > button {
        background-color: #249653;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        min-height: 44px;
    }

    .stButton > button:hover {
        background-color: #197A42;
        color: white;
    }

    [data-testid="stMetricValue"] {
        color: #249653;
    }

    [data-testid="stMetricLabel"] {
        color: #607568;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🌿 PlantCare AI")

    st.caption(
        "AI-powered plant disease detection"
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.success("EfficientNet-B0")

    st.metric(
        "Test Accuracy",
        "96.54%"
    )

    st.metric(
        "Disease Classes",
        "38"
    )

    st.divider()

    st.markdown("### ⚡ Features")

    st.write("🌱 Disease Detection")
    st.write("🎯 Top-3 Predictions")
    st.write("📊 Confidence Scores")

    if torch.cuda.is_available():
        st.write("⚡ GPU Accelerated")
    else:
        st.write("💻 CPU Inference")


# =========================================================
# HEADER
# =========================================================

st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image and let AI identify "
    "possible diseases in seconds."
)

st.caption(
    "Powered by EfficientNet-B0 • 38 plant disease classes"
)


# =========================================================
# MODEL STATS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🎯 Test Accuracy",
        "96.54%"
    )

with col2:
    st.metric(
        "🌱 Classes",
        "38"
    )

with col3:
    st.metric(
        "🧠 Architecture",
        "EfficientNet-B0"
    )

with col4:
    st.metric(
        "⚡ Device",
        "GPU" if torch.cuda.is_available()
        else "CPU"
    )


st.divider()


# =========================================================
# UPLOAD
# =========================================================

st.header("📤 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# WHEN IMAGE IS UPLOADED
# =========================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # -----------------------------------------------------
    # IMAGE + ANALYZE
    # -----------------------------------------------------

    image_col, action_col = st.columns(
        [1.2, 0.8],
        gap="large"
    )


    with image_col:

        st.subheader("🖼️ Your Leaf")

        st.image(
            image,
            use_container_width=True
        )


    with action_col:

        st.subheader("🔍 Analyze")

        st.write(
            "The trained EfficientNet-B0 model "
            "is ready to analyze this image."
        )

        st.write("")

        predict_button = st.button(
            "🌿 Detect Disease",
            use_container_width=True
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    if predict_button:

        with st.spinner(
            "Analyzing leaf..."
        ):

            results = predict(image)


        best_name = results[0][0]
        best_confidence = results[0][1]


        st.divider()


        # =================================================
        # MAIN RESULT
        # =================================================

        st.header("🎯 Prediction Result")

        result_col1, result_col2 = st.columns(
            [2, 1]
        )


        with result_col1:

            st.success(
                f"Predicted Condition\n\n"
                f"**{clean_name(best_name)}**"
            )


        with result_col2:

            st.metric(
                "Confidence",
                f"{best_confidence:.2f}%"
            )


        # =================================================
        # HEALTH / DISEASE STATUS
        # =================================================

        if "healthy" in best_name.lower():

            st.success(
                "🌱 The model predicts that this "
                "plant leaf is healthy."
            )

        else:

            st.warning(
                "⚠️ A possible disease was detected. "
                "Consider inspecting the plant further."
            )


        # =================================================
        # TOP 3
        # =================================================

        st.header("🏆 Top 3 Predictions")


        for rank, (name, confidence) in enumerate(
            results,
            start=1
        ):

            left, right = st.columns(
                [5, 1]
            )


            with left:

                st.write(
                    f"**{rank}. {clean_name(name)}**"
                )

                st.progress(
                    min(
                        confidence / 100,
                        1.0
                    )
                )


            with right:

                st.write(
                    f"**{confidence:.2f}%**"
                )


# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.header("⚡ How It Works")


step1, step2, step3 = st.columns(3)


with step1:

    st.subheader("1️⃣ Upload")

    st.write(
        "Upload a clear image of a plant leaf."
    )


with step2:

    st.subheader("2️⃣ Analyze")

    st.write(
        "EfficientNet-B0 extracts important "
        "visual features from the image."
    )


with step3:

    st.subheader("3️⃣ Predict")

    st.write(
        "The model predicts the disease and "
        "provides confidence scores."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🌿 PlantCare AI  •  "
    "PyTorch + EfficientNet-B0 + Streamlit"
)