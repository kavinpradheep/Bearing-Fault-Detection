import streamlit as st
import numpy as np
from scipy.stats import skew, kurtosis
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
from io import BytesIO
import openpyxl

# Page configuration
st.set_page_config(
    page_title="Bearing Fault Detection",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .company-name {
        font-size: 56px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        color: #2C3E50;
        font-family: 'Arial', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .separator-line {
        height: 3px;
        background: linear-gradient(90deg, #2C3E50, #3498DB, #2C3E50);
        margin: 10px auto 30px auto;
        width: 80%;
        border-radius: 2px;
    }
    .main-header {
        font-size: 38px;
        font-weight: bold;
        color: #34495E;
        margin-bottom: 20px;
        text-align: center;
    }
    .sub-header {
        font-size: 26px;
        font-weight: bold;
        color: #333;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #f8f9fa; /* Light theme default */
        color: #333; /* Dark text for light background */
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
    /* Dark theme adjustments */
    @media (prefers-color-scheme: dark) {
        .info-box {
            background-color: #2c2f33; /* Dark theme background */
            color: #e0e0e0; /* Light text for dark background */
        }
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 15px; 
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 3px solid #28a745;
    }
    .step-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 3px solid #fd7e14;
    }
</style>
""", unsafe_allow_html=True)

# Add company name and separator line before main header
st.markdown('<div class="company-name">Sri Madhura Engineering</div>', unsafe_allow_html=True)
st.markdown('<div class="separator-line"></div>', unsafe_allow_html=True)

# Load both models, scaler, and label encoder
@st.cache_resource
def load_artifacts():
    try:
        lstm_model = load_model("lstm_fault_detection_model.h5")
        cnn_lstm_model = load_model("cnn_lstm_fault_detection_model.h5")
        scaler = joblib.load("scaler.save")
        label_encoder = joblib.load("label_encoder.save")
        return lstm_model, cnn_lstm_model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None

lstm_model, cnn_lstm_model, scaler, label_encoder = load_artifacts()
models_loaded = lstm_model is not None and cnn_lstm_model is not None

# MongoDB Atlas setup
@st.cache_resource
def init_mongodb():
    try:
        # MongoDB Atlas connection string
        connection_string = "mongodb+srv://dinesh:6650@cluster0.slrycce.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
        # For secure credential handling, use Streamlit secrets or environment variables (uncomment as needed):
        # connection_string = st.secrets["mongodb"]["connection_string"]  # Use with secrets.toml
        # from dotenv import load_dotenv; load_dotenv(); connection_string = os.getenv("MONGODB_URI")  # Use with .env

        # Initialize MongoDB client
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        
        # Test the connection
        client.admin.command('ping')
        
        # Access the database
        db = client["bearing_fault_detection"]
        
        #st.success("Successfully connected to MongoDB Atlas!")
        return db
    
    except ConnectionFailure as e:
        st.error(f"Failed to connect to MongoDB Atlas: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while connecting to MongoDB Atlas: {e}")
        return None

# Initialize MongoDB and create upload directory
db = init_mongodb()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Feature extraction function
def extract_features(data):
    max_val = np.max(data)
    min_val = np.min(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    rms = np.sqrt(np.mean(data**2))
    skewness = skew(data)
    kurt = kurtosis(data)
    crest_factor = max_val / rms if rms != 0 else 0
    form_factor = rms / abs(mean_val) if mean_val != 0 else 0
    return [max_val, min_val, mean_val, std_val, rms, skewness, kurt, crest_factor, form_factor]

# Prediction function for both models
def predict_fault(data, model, model_name):
    features = extract_features(data)
    features_scaled = scaler.transform([features])
    features_scaled = features_scaled.reshape((1, 1, -1))  # Assuming both models expect same input shape
    prediction = model.predict(features_scaled, verbose=0)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0], features, prediction

# Process audio input
def process_audio(audio_file):
    try:
        signal, sr = librosa.load(audio_file)
        if len(signal) < 20480:
            st.warning("Audio signal is too short. Padding with zeros.")
            signal = np.pad(signal, (0, 20480 - len(signal)), mode='constant')
        elif len(signal) > 20480:
            signal = signal[:20480]  # Truncate to 20480 samples
        return signal
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Save prediction to MongoDB
def save_prediction(file, input_type, prediction_output):
    try:
        if db is None:
            st.error("Database connection not available")
            return False
            
        # Save file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = file.name
        filename = f"{timestamp}_{original_filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            with open(filepath, 'wb') as f:
                f.write(file.getbuffer())
        except IOError as e:
            st.error(f"❌ Error saving file: {e}")
            return False
            
        # Store metadata in MongoDB
        document = {
            "original_filename": original_filename,
            "saved_filename": filename,
            "filepath": filepath,
            "timestamp": datetime.now(),
            "output": prediction_output,
            "input_type": input_type,
        }
        
        try:
            db.uploads.insert_one(document)
            st.success("✅ Results saved to database successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error saving to database: {e}")
            return False
            
    except Exception as e:
        st.error(f"❌ Error saving results: {e}")
        return False

# Fetch history data from MongoDB
def fetch_history_data():
    if db is None:
        st.error("Database connection not available")
        return None
    try:
        data = list(db.uploads.find({}, {'_id': 0}))
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return None

# Download data as Excel
def download_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='History')
    return output.getvalue()

# UI Components
st.markdown('<div class="main-header">🔧 Bearing Fault Detection System</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    This application uses two deep learning models (LSTM and CNN-LSTM) to detect faults in bearing components based on vibration data.
    Upload a .txt file containing 20480 space-separated acceleration samples or a .wav audio file to diagnose potential bearing issues.
</div>
""", unsafe_allow_html=True)

# Main content in tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Prediction", "ℹ️ Features Explained", "🧠 Models", "📝 Data Collection", "📜 History"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Vibration Data")
        input_type = st.radio("Select input type:", ("Text File (.txt)", "Audio File (.wav)"))
        
        data = None
        if input_type == "Text File (.txt)":
            uploaded_file = st.file_uploader("Choose a .txt file with acceleration data", type="txt")
            if uploaded_file:
                try:
                    content = uploaded_file.read().decode("utf-8")
                    data = np.array(list(map(float, content.strip().split())))
                    if len(data) != 20480:
                        st.warning(f"File contains {len(data)} samples. Expected 20480.")
                    st.success(f"✅ Successfully loaded {len(data)} data points")
                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")
        
        elif input_type == "Audio File (.wav)":
            uploaded_file = st.file_uploader("Choose a .wav file", type="wav")
            if uploaded_file:
                data = process_audio(uploaded_file)
                if data is not None:
                    st.success(f"✅ Loaded audio with {len(data)} samples")
        
        if data is not None:
            # Plot a sample of the data
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data[:1000])  # Plot first 1000 points
            ax.set_title("Sample of Vibration Data")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
            # Add a submit button
            if st.button("🚀 Analyze and Predict Fault"):
                if models_loaded:
                    with st.spinner("Analyzing vibration pattern..."):
                        # Predict with both models
                        lstm_label, lstm_features, lstm_probs = predict_fault(data, lstm_model, "LSTM")
                        cnn_lstm_label, cnn_lstm_features, cnn_lstm_probs = predict_fault(data, cnn_lstm_model, "CNN-LSTM")
                        
                        # Save results to backend
                        if uploaded_file:
                            uploaded_file.seek(0)  # Reset file pointer
                            prediction_output = f"LSTM: {lstm_label}, CNN-LSTM: {cnn_lstm_label}"
                            save_prediction(
                                uploaded_file,
                                input_type,
                                prediction_output
                            )

                        st.markdown("### 🔍 Diagnosis Results")
                        col_lstm, col_cnn_lstm = st.columns(2)
                        
                        with col_lstm:
                            st.markdown(f"""
                            <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center;">
                                <h2 style="color: #155724;">LSTM Model: {lstm_label}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_cnn_lstm:
                            st.markdown(f"""
                            <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center;">
                                <h2 style="color: #155724;">CNN-LSTM Model: {cnn_lstm_label}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show feature values (same for both since input is identical)
                        st.markdown("### 📊 Extracted Features")
                        feature_names = ["Maximum Value", "Minimum Value", "Mean", "Standard Deviation", 
                                        "RMS", "Skewness", "Kurtosis", "Crest Factor", "Form Factor"]
                        features_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Value": lstm_features
                        })
                        st.table(features_df)
                        
                        # Show prediction confidence for both models in one graph
                        st.markdown("### 📈 Prediction Confidence Comparison")
                        class_names = label_encoder.classes_
                        
                        # Create a single comparative bar graph
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bar_width = 0.35
                        index = np.arange(len(class_names))
                        
                        lstm_probs_df = pd.DataFrame({
                            "Fault Type": class_names,
                            "Confidence": lstm_probs[0] * 100
                        })
                        cnn_lstm_probs_df = pd.DataFrame({
                            "Fault Type": class_names,
                            "Confidence": cnn_lstm_probs[0] * 100
                        })
                        
                        ax.bar(index, lstm_probs_df["Confidence"], bar_width, label='LSTM', color='#1f77b4')
                        ax.bar(index + bar_width, cnn_lstm_probs_df["Confidence"], bar_width, label='CNN-LSTM', color='#ff7f0e')
                        
                        ax.set_xlabel("Fault Type")
                        ax.set_ylabel("Confidence (%)")
                        ax.set_title("Prediction Confidence by Model")
                        ax.set_xticks(index + bar_width / 2)
                        ax.set_xticklabels(class_names, rotation=45, ha='right')
                        ax.legend()
                        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.error("One or both models not loaded. Please check the model files.")
    
    with col2:
        st.markdown("### Interpretation Guide")
        st.info("""
        **What This Means:**
        
        The system analyzes vibration patterns using two models to identify bearing conditions:
        
        - **Normal**: No signs of bearing damage or wear
        - **Inner Race Fault**: Damage or defect on the inner raceway
        - **Outer Race Fault**: Damage or defect on the outer raceway
        - **Ball Fault**: Damage on the rolling elements
        - **Combination Fault**: Multiple fault types present simultaneously
        
        Differences between LSTM and CNN-LSTM predictions may arise due to model sensitivities. High confidence scores indicate reliable predictions.
        """)
        
        st.warning("""
        **Recommended Actions:**
        
        - If both models predict 'Normal': Continue regular maintenance
        - If either model detects a fault: Inspect the bearing
        - If both models detect the same fault: Immediate inspection recommended
        - For 'Combination Fault': Urgent attention required
        
        Always verify results with physical inspection.
        """)

with tab2:
    st.markdown('<div class="sub-header">Statistical Features for Fault Detection</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Both models base their predictions on nine key statistical features extracted from the raw vibration data:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 1. Maximum Value")
        st.markdown("The highest amplitude in the vibration signal. High values may indicate impact events.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 2. Minimum Value")
        st.markdown("The lowest amplitude in the vibration signal. Helps establish the signal range.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 3. Mean")
        st.markdown("Average of all signal points. Shifts from zero can indicate sensor bias or load changes.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 4. Standard Deviation")
        st.markdown("Measures signal variability. Increases typically indicate developing faults.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 5. RMS (Root Mean Square)")
        st.markdown("Measures the signal's energy content. Excellent indicator of overall vibration severity.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 6. Skewness")
        st.markdown("Measures asymmetry of vibration distribution. Impulsive faults create positive skewness.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 7. Kurtosis")
        st.markdown("Measures 'peakedness' of the signal. High values indicate impulsive behavior from faults.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 8. Crest Factor")
        st.markdown("Ratio of peak value to RMS. Increases in early stages of bearing damage.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 9. Form Factor")
        st.markdown("Ratio of RMS to mean absolute value. Sensitive to signal shape changes.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Feature Importance
    
    These features work together to characterize different fault patterns:
    
    - **Inner race faults** typically show high kurtosis and crest factor with periodic impulses
    - **Outer race faults** create distinct impacts at specific frequencies
    - **Ball faults** generate more chaotic patterns with higher frequency components
    - **Combined faults** show complex mixtures of these characteristics
    """)

with tab3:
    st.markdown('<div class="sub-header">Neural Network Architectures</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### Model Architectures
        
        This application uses two neural networks for fault detection:
        
        #### 1. LSTM Model
        The Long Short-Term Memory (LSTM) model is designed for sequential data analysis, detecting temporal patterns in vibration signals.
        
        - **Input Layer**: 9 statistical features
        - **LSTM Layers**: Multiple layers to learn temporal dependencies
        - **Dropout Layers**: Prevent overfitting
        - **Dense Layers**: Interpret patterns
        - **Output Layer**: Probabilities for each fault class
        
        #### 2. CNN-LSTM Model
        The CNN-LSTM model combines Convolutional Neural Networks (CNN) with LSTM to extract spatial and temporal features.
        
        - **Input Layer**: 9 statistical features
        - **CNN Layers**: Extract spatial patterns
        - **LSTM Layers**: Capture temporal dependencies
        - **Dense Layers**: Classify faults
        - **Output Layer**: Probabilities for each fault class
        
        #### Key Advantages:
        - **LSTM**: Excels at capturing progressive deterioration
        - **CNN-LSTM**: Combines spatial feature extraction with temporal analysis
        - Both are robust to noise and detect early-stage faults
        """)
    
    with col2:
        st.image("https://miro.medium.com/max/1400/1*qn9Xl6PsitXUXP3PcY9Jyw.png", caption="LSTM Cell Architecture")
        
    st.markdown("""
    ### Training Process
    
    Both models were trained on thousands of vibration samples:
    
    1. **Data Collection**: Vibration data from bearings in various conditions
    2. **Feature Extraction**: Time-domain features calculated
    3. **Data Augmentation**: Added controlled noise
    4. **Model Training**: Backpropagation through time
    5. **Validation**: Tested on separate datasets
    6. **Fine-Tuning**: Optimized hyperparameters
    
    Both models achieve over 92% accuracy on unseen test data.
    """)

with tab4:
    st.markdown('<div class="sub-header">Data Collection Methodology</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Step-by-Step Vibration Data Collection Process

    High-quality data acquisition is essential for accurate fault prediction:
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### Step 1: Sensor Placement")
        st.markdown("""
        - Attach accelerometer to bearing housing
        - Use adhesive or magnetic mounts
        - Place near load zone
        - Single-axis or tri-axial sensor
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### Step 2: Analog-to-Digital Conversion")
        st.markdown("""
        - Connect to ADC/DAQ system
        - Sampling rate ≥20 kHz
        - Apply anti-aliasing filters
        - Choose appropriate voltage range
        - Ensure low-noise wiring
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### Step 3: Operating Conditions")
        st.markdown("""
        - Operate under normal conditions
        - Warm up system
        - Record RPM and ambient conditions
        - Ensure steady-state
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### Step 4: Sample Recording")
        st.markdown("""
        - Capture 20,480 samples
        - Acceleration magnitude
        - Use float format
        - Store as .txt with space-separated values
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### Step 5: Preprocessing")
        st.markdown("""
        - Convert voltage to acceleration
        - Ensure unfiltered data
        - Verify sample count
        - Confirm clean data
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown("#### Step 6: Validation & Upload")
        st.markdown("""
        - Inspect waveform
        - Validate sample count
        - Upload for prediction
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Hardware Guidelines

    - **Accelerometer**: MEMS or piezoelectric (20 kHz+)
    - **DAQ/ADC**: 16-bit resolution, ≥20 kHz

    ### Applicable Bearing Types

    - Deep groove ball bearings
    - Cylindrical/tapered roller bearings
    - Angular contact/thrust bearings
    """)

with tab5:
    st.markdown('<div class="sub-header">Analysis History</div>', unsafe_allow_html=True)
    
    # Fetch historical data
    history_df = fetch_history_data()
    
    if history_df is not None and not history_df.empty:
        # Add download button
        excel_data = download_excel(history_df)
        st.download_button(
            label="📥 Download History as Excel",
            data=excel_data,
            file_name="bearing_analysis_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Display history table with formatting
        st.markdown("### Previous Analyses")
        
        # Format timestamp
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Reorder and rename columns for display
        display_columns = {
            'timestamp': 'Date & Time',
            'input_type': 'Input Type',
            'original_filename': 'File Name',
            'output': 'Prediction Results'
        }
        
        display_df = history_df[display_columns.keys()].rename(columns=display_columns)
        
        # Display table with custom styling
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No analysis history available yet.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Bearing Fault Detection System v1.2 | Developed with TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)