<h1 align="center"> Preprocessor – Intelligent Data Preprocessing Advisor</h1>

<p align="center">
  <b>Preprocessor</b> is a Python library that analyzes your dataset and suggests the right preprocessing techniques —<br>
  for <b>missing values</b>, <b>encoding</b>, <b>transformation</b>, and <b>scaling</b> —<br>
  based on data characteristics and the <b>ML model</b> you plan to use.
</p>

Instead of blindly applying preprocessing steps, Preprocessor thinks like a data scientist and tells you:
- What to do
- Why to do it
- When to do it
- What to avoid

### 🚀 Why Preprocessor?

Data preprocessing is order-sensitive and model-dependent.
Doing the wrong step at the wrong time can silently ruin model performance.

Preprocessor helps you avoid common mistakes like:
- Scaling categorical features
- Transforming binary variables
- Encoding target variables inside features
- Applying scaling before handling missing values
- Using transformations with tree-based models

### ✨ Key Features:

**✅ DataType Correctness**

**✅ Missing Value Analysis**
- Detects missing values
- Suggests mean / median / mode strategies
- Explains why a strategy is recommended

**✅ Transformation Suggestions**
- Detects skewness
- Recommends log, sqrt, or Yeo-Johnson only when necessary
- Skips: binary numeric features, ID-like columns, tree-based models

**✅ Encoding Guidance**
- Binary → Label Encoding
- Low-cardinality categorical → One-Hot Encoding
- High-cardinality categorical → Frequency Encoding
- Prevents: encoding numeric features, encoding target variable accidentally

**✅ Scaling Recommendations**
- Model-aware (Linear vs Tree-based)
- Chooses between: StandardScaler, RobustScaler
- Scales ONLY original continuous numeric features
- Never scales: one-hot encoded columns, binary indicators

**✅ Correct Execution Order (Guaranteed):**

1️⃣ DataType Correctness

2️⃣ Missing Values

3️⃣ Transformation

4️⃣ Encoding

5️⃣ Scaling

 ### 🎥 Demo Video:

Watch how **Preprocessor** intelligently suggests preprocessing steps:

▶️ https://github.com/KashishPundir/Preprocessor/blob/main/demo/Demo%20Video%20of%20Cognia%20and%20Preprocessor%20Library.mp4


### 📂 Project Structure:

```
PREPROCESSOR/
│
├── demo/
│   └── run_pipeline.py        # Example usage
│
├──
│   └── preprocessor/
│       ├── __init__.py
│       ├── combine_all.py     # Main pipeline
│       ├── missing.py         # Missing value suggestions
│       ├── transformation.py  # Skew & transformation logic
│       ├── encoding.py        # Encoding suggestions
│       ├── scaling.py         # Model-aware scaling
│       └── printers.py        # Pretty report printing
│       └── datatype_fixer.py  # datatype correctness suggestion
│
├── pyproject.toml
├── requirements.txt
├── README.md
```

### ⚙️ Installation:
1️⃣ Clone the repository
```
git clone https://github.com/your-username/preprocessor.git
cd preprocessor
```

2️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 📌 When to Use Preprocessor?

Use this library before model training, when you want:
- clarity on preprocessing choices
- justification for each step
- confidence that your pipeline is correct

### ⭐ Support

If this project helps you:

⭐ Star the repository

🐛 Report issues

💡 Suggest improvements



