#!/usr/bin/env python3
"""
Test script for EduFrame AI lightweight models
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import pickle
from pathlib import Path
from transformers import DistilBertModel, DistilBertTokenizer
import torch

sys.path.append(str(Path(__file__).parent.parent))

class ModelTester:
    def __init__(self):
        self.models_loaded = False
        self.fln_model = None
        self.fln_scaler = None
        self.nlp_tokenizer = None
        self.nlp_model = None
        self.success_model = None
    
    def load_models(self):
        """Load all models into memory"""
        print("📥 Loading models...")
        
        try:
            # Load FLN predictor
            self.fln_model = joblib.load('models/fln_predictor.joblib')
            self.fln_scaler = joblib.load('models/fln_scaler.joblib')
            print("✅ FLN model loaded")
            
            # Load NLP designer
            nlp_dir = 'models/nlp_designer'
            if os.path.exists(nlp_dir):
                self.nlp_tokenizer = DistilBertTokenizer.from_pretrained(nlp_dir)
                self.nlp_model = DistilBertModel.from_pretrained(nlp_dir)
                print("✅ NLP model loaded")
            
            # Load success predictor
            with open('models/success_predictor.pkl', 'rb') as f:
                self.success_model = pickle.load(f)
            print("✅ Success predictor loaded")
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False
    
    def test_fln_predictor(self):
        """Test FLN predictor with sample input"""
        print("\n🧪 Testing FLN Predictor...")
        
        if not self.models_loaded:
            print("❌ Models not loaded")
            return None
        
        # Sample input (10 features as expected by model)
        sample_input = np.random.randn(1, 10)
        
        # Scale input
        scaled_input = self.fln_scaler.transform(sample_input)
        
        # Make prediction
        prediction = self.fln_model.predict(scaled_input)[0]
        
        print(f"📊 Sample prediction: {prediction:.3f}")
        print(f"✅ FLN Predictor working!")
        
        return prediction
    
    def test_nlp_designer(self, text):
        """Test NLP designer with text input"""
        print(f"\n🧪 Testing NLP Designer with: '{text}'")
        
        if not self.models_loaded or self.nlp_tokenizer is None:
            print("❌ NLP model not loaded")
            return None
        
        # Tokenize input
        inputs = self.nlp_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.nlp_model(**inputs)
        
        # Get sentence embedding (mean of last hidden state)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        print(f"📊 Input processed: {len(text)} characters")
        print(f"📊 Embedding shape: {embeddings.shape}")
        print(f"✅ NLP Designer working!")
        
        # Simulate program recommendation based on keywords
        recommendations = self._generate_recommendation(text)
        return recommendations
    
    def _generate_recommendation(self, text):
        """Generate mock recommendations based on text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['reading', 'literacy', 'book']):
            return {
                'program': 'Foundational Literacy Program',
                'success_prob': 0.85,
                'duration': '6 months',
                'key_activities': ['Phonics training', 'Reading circles', 'Library setup']
            }
        elif any(word in text_lower for word in ['math', 'numeracy', 'calculation']):
            return {
                'program': 'Numeracy Intervention',
                'success_prob': 0.78,
                'duration': '8 months',
                'key_activities': ['Math games', 'Problem-solving workshops', 'Math kits']
            }
        elif any(word in text_lower for word in ['dropout', 'retention', 'attendance']):
            return {
                'program': 'Student Retention Program',
                'success_prob': 0.65,
                'duration': '12 months',
                'key_activities': ['Mentorship', 'Parent engagement', 'Scholarships']
            }
        else:
            return {
                'program': 'Holistic Learning Program',
                'success_prob': 0.72,
                'duration': '10 months',
                'key_activities': ['Teacher training', 'Learning materials', 'Assessment system']
            }
    
    def test_success_predictor(self, program_features=None):
        """Test success predictor"""
        print("\n🧪 Testing Success Predictor...")
        
        if not self.models_loaded:
            print("❌ Models not loaded")
            return None
        
        # Use provided features or generate sample
        if program_features is None:
            program_features = {
                'budget_adequacy': np.random.uniform(0.3, 0.9),
                'teacher_training': np.random.uniform(0.4, 0.95),
                'stakeholder_support': np.random.uniform(0.5, 0.8),
                'implementation_timeline': np.random.uniform(0.6, 1.0),
                'previous_success_rate': np.random.uniform(0.0, 1.0)
            }
        
        # Convert to array for prediction
        features_array = np.array([list(program_features.values())])
        
        # Make prediction
        success_prob = self.success_model.predict(features_array)[0]
        success_prob = max(0, min(1, success_prob))  # Clip to 0-1
        
        print("📊 Program Features:")
        for key, value in program_features.items():
            print(f"   {key.replace('_', ' ').title()}: {value:.1%}")
        
        print(f"\n🎯 Predicted Success Probability: {success_prob:.1%}")
        
        # Add interpretation
        if success_prob > 0.8:
            interpretation = "HIGH SUCCESS LIKELY 🎉"
        elif success_prob > 0.6:
            interpretation = "MODERATE SUCCESS LIKELY ⚡"
        else:
            interpretation = "NEEDS IMPROVEMENT 🔧"
        
        print(f"📈 Interpretation: {interpretation}")
        print("✅ Success Predictor working!")
        
        return {
            'success_probability': success_prob,
            'features': program_features,
            'interpretation': interpretation
        }
    
    def test_integrated_workflow(self):
        """Test complete workflow from problem to success prediction"""
        print("\n" + "=" * 60)
        print("🎮 TESTING COMPLETE WORKFLOW")
        print("=" * 60)
        
        # Get user input
        problem = input("\n📝 Enter an educational problem: ")
        
        print("\n🔍 Step 1: Analyzing problem with NLP...")
        nlp_result = self.test_nlp_designer(problem)
        
        print("\n🔍 Step 2: Generating program recommendation...")
        if nlp_result:
            print(f"   📋 Program: {nlp_result['program']}")
            print(f"   ⏱️ Duration: {nlp_result['duration']}")
            print(f"   🎯 Activities: {', '.join(nlp_result['key_activities'][:2])}...")
        
        print("\n🔍 Step 3: Predicting FLN improvement...")
        fln_prediction = self.test_fln_predictor()
        if fln_prediction:
            improvement = fln_prediction * 100
            print(f"   📈 Expected FLN improvement: {improvement:.1f}%")
        
        print("\n🔍 Step 4: Predicting overall success...")
        success_result = self.test_success_predictor()
        
        print("\n" + "=" * 60)
        print("📊 WORKFLOW COMPLETE!")
        print("=" * 60)
        
        if success_result and nlp_result:
            final_score = (success_result['success_probability'] + nlp_result['success_prob']) / 2
            
            print(f"\n🎯 Final Combined Score: {final_score:.1%}")
            print(f"🏆 Program: {nlp_result['program']}")
            print(f"📈 Success Range: {min(success_result['success_probability'], nlp_result['success_prob']):.1%} - "
                  f"{max(success_result['success_probability'], nlp_result['success_prob']):.1%}")
            
            if final_score > 0.7:
                print("\n✅ RECOMMENDATION: IMPLEMENT THIS PROGRAM!")
            else:
                print("\n⚠️ RECOMMENDATION: REVISE PROGRAM DESIGN")

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧠 EDUFRAME AI - LIGHTWEIGHT MODEL TESTING")
    print("=" * 60)
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("❌ Models directory not found!")
        print("Run: python create_models.py first")
        return
    
    # Initialize tester
    tester = ModelTester()
    
    # Load models
    if not tester.load_models():
        print("❌ Failed to load models")
        return
    
    print("\n" + "=" * 60)
    print("🧪 RUNNING INDIVIDUAL MODEL TESTS")
    print("=" * 60)
    
    # Test individual models
    fln_result = tester.test_fln_predictor()
    
    # Test NLP with sample input
    sample_problems = [
        "Grade 3 students struggling with reading comprehension",
        "High dropout rates in rural secondary schools",
        "Poor math performance in tribal areas"
    ]
    
    for problem in sample_problems:
        nlp_result = tester.test_nlp_designer(problem)
    
    # Test success predictor
    success_result = tester.test_success_predictor()
    
    print("\n" + "=" * 60)
    print("✅ ALL MODEL TESTS COMPLETED")
    print("=" * 60)
    
    # Ask user if they want to test complete workflow
    response = input("\n🎮 Test complete workflow with your own input? (y/n): ")
    if response.lower() == 'y':
        tester.test_integrated_workflow()
    
    # Storage summary
    print("\n" + "=" * 60)
    print("💾 STORAGE SUMMARY")
    print("=" * 60)
    
    total_size = 0
    for root, dirs, files in os.walk("models"):
        for file in files:
            file_path = os.path.join(root, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size += size_mb
            if size_mb > 10:
                print(f"📁 {file}: {size_mb:.1f} MB")
    
    print(f"\n📊 Total models size: {total_size:.1f} MB")
    print(f"✅ Git LFS compatible: {'YES' if total_size < 5000 else 'WARNING >5GB'}")
    
    print("\n🎉 Day 1 Complete! Models are ready for Day 2 development.")

if __name__ == "__main__":
    main()