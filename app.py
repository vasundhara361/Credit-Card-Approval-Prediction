from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open("credit_card_model.pkl", "rb") as f:
    model = pickle.load(f)
    print(type(model))
print(model)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = pd.DataFrame([{
        "Age": int(request.form["age"]),
        "Gender": request.form["gender"],
        "Annual_Income": int(request.form["income"]),
        "Employment_Years": int(request.form["employment"]),
        "Credit_Score": int(request.form["score"]),
        "Existing_Loans": int(request.form["loans"]),
        "Loan_Amount": int(request.form["loan_amount"]),
        "Marital_Status": request.form["marital"],
        "Education": request.form["education"],
        "Property_Owner": request.form["property"],
        "Default_History": request.form["default"]
    }])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "✅ Credit Card Approved"
    else:
        result = "❌ Credit Card Rejected"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)