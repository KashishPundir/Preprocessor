🧠 Preprocessor – Intelligent Data Preprocessing Advisor

Preprocessor is a Python library that analyzes your dataset and suggests the right preprocessing techniques —
for missing values, encoding, transformation, and scaling —
based on data characteristics and the ML model you plan to use.

Instead of blindly applying preprocessing steps, Preprocessor thinks like a data scientist and tells you:

What to do

Why to do it

When to do it

What to avoid

🚀 Why Preprocessor?

Data preprocessing is order-sensitive and model-dependent.
Doing the wrong step at the wrong time can silently ruin model performance.

Preprocessor helps you avoid common mistakes like:

Scaling categorical features

Transforming binary variables

Encoding target variables inside features

Applying scaling before handling missing values

Using transformations with tree-based models

✨ Key Features

✅ Missing Value Analysis

Detects missing values

Suggests mean / median / mode strategies

Explains why a strategy is recommended

✅ Transformation Suggestions

Detects skewness

Recommends log, sqrt, or Yeo-Johnson only when necessary

Skips:

binary numeric features

ID-like columns

tree-based models

✅ Encoding Guidance

Binary → Label Encoding

Low-cardinality categorical → One-Hot Encoding

High-cardinality categorical → Frequency Encoding

Prevents:

encoding numeric features

encoding target variable accidentally

✅ Scaling Recommendations

Model-aware (Linear vs Tree-based)

Chooses between:

StandardScaler

RobustScaler

Scales ONLY original continuous numeric features

Never scales:

one-hot encoded columns

binary indicators

✅ Correct Execution Order (Guaranteed)

1️⃣ Missing Values
2️⃣ Transformation
3️⃣ Encoding
4️⃣ Scaling

 ## 🎥 Demo Video

Watch how **Preprocessor** intelligently suggests preprocessing steps:

▶️ https://user-images.githubusercontent.com/xxxx/preprocessor-demo.mp4


📂 Project Structure
PREPROCESSOR/
│
├── demo/
│   └── run_pipeline.py        # Example usage
│
├── src/
│   └── preprocessor/
│       ├── __init__.py
│       ├── combine_all.py     # Main pipeline
│       ├── missing.py         # Missing value suggestions
│       ├── transformation.py  # Skew & transformation logic
│       ├── encoding.py        # Encoding suggestions
│       ├── scaling.py         # Model-aware scaling
│       └── printers.py        # Pretty report printing
│
├── pyproject.toml
├── requirements.txt
├── README.md

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/preprocessor.git
cd preprocessor

2️⃣ Install dependencies
pip install -r requirements.txt

📌 When to Use Preprocessor

Use this library before model training, when you want:

clarity on preprocessing choices

justification for each step

confidence that your pipeline is correct

⭐ Support

If this project helps you:

⭐ Star the repository

🐛 Report issues

💡 Suggest improvements

