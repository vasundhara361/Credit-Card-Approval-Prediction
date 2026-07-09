# Credit Card Approval Prediction

## Project Description
This project predicts whether a credit card application will be approved or rejected using Machine Learning.

## Technologies Used
- Python
- Flask
- Decision Tree Classifier
- Pandas
- Scikit-learn
- HTML
- CSS

## Dataset Features
- Age
- Gender
- Annual Income
- Employment Years
- Credit Score
- Existing Loans
- Loan Amount
- Marital Status
- Education
- Property Owner
- Default History

## How to Run

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the Flask application:

```bash
python app.py
```

3. Open your browser and visit:

```
http://127.0.0.1:5000
```

## Project Structure

```
vasu/
│
├── app.py
├── credit_card_model.pkl
├── credit_card_dataset.csv
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── style.css
```