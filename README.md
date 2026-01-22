# EduFrame-AI
# EduFrame AI - Lightweight Models Version

## 🎯 Alternative Models (No TensorFlow/CatBoost)
| Model | Original | Alternative | Size |
|-------|----------|-------------|------|
| FLN Predictor | TensorFlow | **RandomForest** | ~3MB |
| NLP Designer | Large Model | **DistilBERT** | ~267MB |
| Success Predictor | CatBoost | **LightGBM** | ~2MB |
| **Total** | **~7GB** | **~272MB** | ✅ |

## 🚀 Quick Start
```bash
# 1. Clone with Git LFS
git lfs install
git clone https://github.com/yourusername/eduframe-ai.git
cd eduframe-ai

# 2. Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Create models (if not already in repo)
python create_models.py

# 4. Test models
python tests/test_models.py

# 5. Run interactive UI
streamlit run test_ui.py
