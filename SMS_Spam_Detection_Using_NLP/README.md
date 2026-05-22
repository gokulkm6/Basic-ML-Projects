
# 📩 SMS Spam Detection using Logistic Regression

This project demonstrates a simple yet effective approach to **detecting spam messages** using **Natural Language Processing (NLP)** and **Logistic Regression**. The model classifies SMS text messages as either **spam (0)** or **ham (1)** based on their content.

---

## 🗂️ Dataset

- Source: `spam.csv`
- Encoding: `latin-1`
- Preprocessing Steps:
  - Removed unused columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`)
  - Removed duplicate entries
  - Renamed columns to `['spam/ham', 'sms']`
  - Label Mapping:
    - `'spam'` → `0`
    - `'ham'` → `1`

---

## 🛠️ Libraries Used

- `pandas`, `numpy` for data handling
- `scikit-learn` for model building and evaluation
- `nltk` for stopword filtering (optional)
- `TfidfVectorizer` for feature extraction from text

---

## ⚙️ Model Workflow

1. **Data Loading**  
2. **Cleaning and Preprocessing**
3. **TF-IDF Vectorization**:
   - Removes stopwords
   - Converts text to lowercase
4. **Model Training**:
   - Algorithm: `LogisticRegression`
5. **Evaluation**:
   - Accuracy Score on Test Data

---

## 🧪 Example Output

```bash
Accuracy Score: 0.976
```

This indicates the model correctly classifies approximately 97.6% of SMS messages.

---

## 🚀 How to Run

1. Make sure you have the dependencies installed:

```bash
pip install pandas numpy scikit-learn nltk
```

2. Run the script:

```bash
python sms_spam_detection.py
```

---

## 📌 Notes

- The model uses a very lightweight and interpretable approach.
- TF-IDF ensures that word frequency is normalized and informative.
- No deep learning or GPU is required – runs in seconds.

---

## 📂 File Structure

```
📦 SpamDetectionProject
├── spam.csv
├── sms_spam_detection.py
└── README.md
```

---

## 🧠 Future Improvements

- Add advanced NLP preprocessing (lemmatization, POS tagging)
- Experiment with other models like Naive Bayes, SVM
- Build a web app using Streamlit or Flask for real-time SMS classification

---

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).

