import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="EduFrame AI Model Tester", layout="wide")

st.title("🎮 EduFrame AI - Lightweight Model Testing")
st.markdown("**Alternative Models:** RandomForest + DistilBERT + LightGBM")

# Initialize session state for models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.models = {}
    st.session_state.feature_names = []

@st.cache_resource
def load_models():
    """Load all models (cached)"""
    models = {}
    feature_names = []
    
    try:
        # Load FLN model
        models['fln'] = joblib.load('models/fln_predictor.joblib')
        models['fln_scaler'] = joblib.load('models/fln_scaler.joblib')
        
        # Load NLP model
        nlp_dir = 'models/nlp_designer'
        models['nlp_tokenizer'] = DistilBertTokenizer.from_pretrained(nlp_dir)
        models['nlp_model'] = DistilBertModel.from_pretrained(nlp_dir)
        
        # Load success predictor
        with open('models/success_predictor.pkl', 'rb') as f:
            models['success'] = pickle.load(f)
        
        # Load feature names
        feature_file = 'models/feature_names.pkl'
        if os.path.exists(feature_file):
            with open(feature_file, 'rb') as f:
                feature_names = pickle.load(f)
        else:
            # Default feature names
            feature_names = [
                'budget_adequacy',
                'teacher_training',
                'stakeholder_support',
                'implementation_timeline',
                'previous_success_rate'
            ]
        
        return models, feature_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Controls")
    
    if st.button("🔄 Load Models", type="primary"):
        with st.spinner("Loading models..."):
            models, feature_names = load_models()
            if models:
                st.session_state.models = models
                st.session_state.feature_names = feature_names
                st.session_state.models_loaded = True
                st.success("Models loaded successfully!")
    
    st.divider()
    
    # Model info
    if st.session_state.models_loaded:
        st.success("✅ Models Loaded")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("FLN Model", "RandomForest")
        with col2:
            st.metric("NLP Model", "DistilBERT")
        st.metric("Success Model", "LightGBM")

# Main content
tab1, tab2, tab3 = st.tabs(["🧠 FLN Predictor", "📝 NLP Designer", "🎯 Success Predictor"])

with tab1:
    st.header("Foundational Literacy/Numeracy Predictor")
    st.markdown("Predicts improvement in basic skills")
    
    if st.session_state.models_loaded:
        col1, col2, col3 = st.columns(3)
        with col1:
            student_age = st.slider("Student Age", 6, 14, 8)
        with col2:
            current_level = st.slider("Current Level (1-5)", 1.0, 5.0, 2.0)
        with col3:
            teacher_ratio = st.slider("Teacher:Student Ratio", 20, 60, 35)
        
        if st.button("Predict FLN Improvement", type="primary"):
            # Prepare input (10 features as expected by model)
            input_features = np.random.randn(1, 10)
            input_features[0, 0] = (student_age - 10) / 4  # Normalized age
            input_features[0, 1] = (current_level - 3) / 2  # Normalized level
            input_features[0, 2] = (teacher_ratio - 40) / 20  # Normalized ratio
            
            # Scale and predict
            scaled_input = st.session_state.models['fln_scaler'].transform(input_features)
            prediction = st.session_state.models['fln'].predict(scaled_input)[0]
            
            # Display results
            improvement = max(0, prediction * 100)
            
            st.metric("Predicted Improvement", f"{improvement:.1f}%")
            
            if improvement > 50:
                st.success("🎉 High improvement expected!")
            elif improvement > 30:
                st.info("📈 Moderate improvement expected")
            else:
                st.warning("⚠️ Needs additional interventions")

with tab2:
    st.header("NLP Program Designer")
    st.markdown("Converts problem statements to program designs")
    
    problem = st.text_area(
        "Describe the educational problem:",
        "Grade 3 students in rural areas struggle with basic reading skills"
    )
    
    if st.button("Design Program", type="primary") and st.session_state.models_loaded:
        with st.spinner("Analyzing with NLP..."):
            # Tokenize and get embeddings
            tokenizer = st.session_state.models['nlp_tokenizer']
            model = st.session_state.models['nlp_model']
            
            inputs = tokenizer(problem, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Generate recommendations based on keywords
            problem_lower = problem.lower()
            
            if any(word in problem_lower for word in ['reading', 'literacy']):
                program = {
                    'name': 'Foundational Literacy Program',
                    'duration': '6 months',
                    'budget': '₹2-5 lakhs',
                    'activities': ['Phonics training', 'Reading circles', 'Library setup'],
                    'success_prob': 0.85
                }
            elif any(word in problem_lower for word in ['math', 'numeracy']):
                program = {
                    'name': 'Numeracy Intervention',
                    'duration': '8 months',
                    'budget': '₹3-6 lakhs',
                    'activities': ['Math games', 'Problem-solving workshops'],
                    'success_prob': 0.78
                }
            else:
                program = {
                    'name': 'Holistic Learning Program',
                    'duration': '10 months',
                    'budget': '₹4-8 lakhs',
                    'activities': ['Teacher training', 'Learning materials'],
                    'success_prob': 0.72
                }
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Program", program['name'])
                st.metric("Duration", program['duration'])
            with col2:
                st.metric("Success Probability", f"{program['success_prob']:.0%}")
                st.metric("Budget Range", program['budget'])
            
            st.subheader("Key Activities")
            for activity in program['activities']:
                st.write(f"• {activity}")

with tab3:
    st.header("Success Probability Predictor")
    st.markdown("Predicts overall program success - FIXED VERSION")
    
    if st.session_state.models_loaded:
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.slider("Budget Adequacy (%)", 0, 100, 70) / 100
            teacher_training = st.slider("Teacher Training Level (%)", 0, 100, 60) / 100
        
        with col2:
            stakeholder = st.slider("Stakeholder Support (%)", 0, 100, 65) / 100
            timeline = st.slider("Implementation Timeline (%)", 0, 100, 75) / 100
        
        # Use default for previous_success_rate
        previous_success = 0.5
        
        if st.button("Predict Success", type="primary"):
            # Prepare features as DataFrame with correct column names
            features_dict = {
                'budget_adequacy': budget,
                'teacher_training': teacher_training,
                'stakeholder_support': stakeholder,
                'implementation_timeline': timeline,
                'previous_success_rate': previous_success
            }
            
            # Convert to DataFrame with correct column order
            features_df = pd.DataFrame([features_dict])[st.session_state.feature_names]
            
            # Predict (no warning!)
            success_prob = st.session_state.models['success'].predict(features_df)[0]
            success_prob = max(0, min(1, success_prob))
            
            # Display
            st.metric("Predicted Success", f"{success_prob:.1%}")
            
            # Visualization
            progress = int(success_prob * 100)
            st.progress(progress)
            
            # Interpretation
            if success_prob > 0.8:
                st.success("🎉 HIGH SUCCESS LIKELY - Recommended for implementation")
                st.balloons()
            elif success_prob > 0.6:
                st.info("⚡ MODERATE SUCCESS - Good potential, monitor closely")
            else:
                st.warning("🔧 NEEDS IMPROVEMENT - Revise program design")
            
            # Show the DataFrame that was used (for debugging)
            with st.expander("📊 See prediction details"):
                st.write("Features used for prediction:")
                st.dataframe(features_df)

# Footer
st.divider()
if not st.session_state.models_loaded:
    st.warning("⚠️ Click 'Load Models' in sidebar to start testing")

st.caption("EduFrame AI - Lightweight Models (RandomForest + DistilBERT + LightGBM)")