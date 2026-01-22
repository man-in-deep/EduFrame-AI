# create_models.py - COMPLETE FILE
import os
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from transformers import DistilBertTokenizer, DistilBertModel

def create_fln_predictor():
    """Create FLN predictor using RandomForest"""
    print("🤖 Creating FLN Predictor (Random Forest)...")
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 10000
    X = np.random.randn(n_samples, 10)
    y = 0.3 + 0.5 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    # Create and train model
    fln_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    fln_model.fit(X, y)
    
    # Save model
    joblib.dump(fln_model, 'models/fln_predictor.joblib')
    
    # Create and save scaler
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, 'models/fln_scaler.joblib')
    
    size_mb = os.path.getsize('models/fln_predictor.joblib') / (1024*1024)
    print(f"✅ FLN Predictor saved: {size_mb:.1f} MB")
    
    return fln_model

def create_nlp_designer():
    """Create NLP designer using DistilBERT"""
    print("\n🤖 Setting up NLP Designer (DistilBERT)...")
    
    # Use small DistilBERT model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    # Save locally
    model_dir = 'models/nlp_designer'
    os.makedirs(model_dir, exist_ok=True)
    
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    
    # Calculate total size
    total_size = sum(os.path.getsize(os.path.join(model_dir, f)) 
                    for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f)))
    
    print(f"✅ NLP Designer saved to: {model_dir}")
    print(f"   Total size: {total_size / (1024*1024):.1f} MB")
    
    return model, tokenizer

def create_success_predictor():
    """Create success predictor using LightGBM"""
    print("\n🤖 Creating Success Predictor (LightGBM)...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'budget_adequacy': np.random.uniform(0.1, 1.0, n_samples),
        'teacher_training': np.random.uniform(0.2, 0.95, n_samples),
        'stakeholder_support': np.random.uniform(0.3, 0.9, n_samples),
        'implementation_timeline': np.random.uniform(0.5, 1.0, n_samples),
        'previous_success_rate': np.random.uniform(0.0, 1.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target
    X = df.values
    y = (0.3 * df['budget_adequacy'] + 
         0.25 * df['teacher_training'] + 
         0.2 * df['stakeholder_support'] + 
         0.15 * df['implementation_timeline'] + 
         0.1 * df['previous_success_rate'] +
         0.05 * np.random.randn(n_samples))
    
    # Train LightGBM model
    success_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        random_state=42
    )
    success_model.fit(X, y)
    
    # Save model
    with open('models/success_predictor.pkl', 'wb') as f:
        pickle.dump(success_model, f)
    
    size_mb = os.path.getsize('models/success_predictor.pkl') / (1024*1024)
    print(f"✅ Success Predictor saved: {size_mb:.1f} MB")
    
    return success_model

def main():
    """Create all models"""
    os.makedirs('models', exist_ok=True)
    
    print("=" * 60)
    print("🤖 CREATING EDUFRAME AI MODELS (Lightweight Version)")
    print("=" * 60)
    
    # Create all models
    fln_model = create_fln_predictor()
    nlp_model, tokenizer = create_nlp_designer()
    success_model = create_success_predictor()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MODEL CREATION SUMMARY")
    print("=" * 60)
    
    total_size = 0
    for root, dirs, files in os.walk('models'):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    print(f"✅ All models created successfully!")
    print(f"📁 Total models size: {total_size / (1024*1024):.1f} MB")
    print(f"🎯 Ready for testing and Git LFS tracking!")
    print("\n🎮 Alternative Models Used:")
    print("   • FLN Predictor: RandomForest (instead of TensorFlow)")
    print("   • NLP Designer: DistilBERT (lightweight transformer)")
    print("   • Success Predictor: LightGBM (instead of CatBoost)")

if __name__ == "__main__":
    main()