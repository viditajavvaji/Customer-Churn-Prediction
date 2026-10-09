# Customer Churn Prediction

## About this project

This project estimates whether a customer of a subscription service may leave. This is called **customer churn**. Losing customers can reduce a business's income, so the app helps a business notice possible churn early and think about ways to keep the customer.

The app is a demonstration of machine learning. Its result is an estimate from a model trained on customer data. It is not a guarantee that a customer will stay or leave.

## What a user can do

When the app opens, the user chooses one of two options:

1. **Check one customer:** enter the customer's details in a form and get a prediction.
2. **Check multiple customers:** upload a CSV file with one customer on each row and get a results table.

The table is kept simple: it shows the customer's name, churn prediction, probability, and risk level. Selecting a row opens the customer's result and generates an AI-written explanation with possible follow-up steps.

The interface also has a dark mode toggle. This changes the page colors to make the app more comfortable to view in a darker setting. The choice is remembered in that browser.

## Customer information used

The model uses 19 details that were also used when it was trained:

- **Customer and household:** gender, senior-citizen status, partner, and dependents.
- **Time with the company:** tenure in months.
- **Services:** phone service, multiple phone lines, internet service, online security, online backup, device protection, technical support, streaming TV, and streaming movies.
- **Billing:** contract type, paperless billing, payment method, monthly charges, and total charges.

The CSV may also have a `Name` column. It is only used to label the result in the table; the model does not use the name to make its prediction. A `Churn` column in an uploaded file is not used to make predictions.

## How a prediction is made

1. The app sends the customer's 19 details to the Python API.
2. The API applies the same data preparation that was saved with the trained model. For example, it prepares number fields and turns service or billing categories into values the model can read.
3. The Logistic Regression model estimates the chance of churn and returns a churn or no-churn result.
4. The app turns that chance into a simple risk band and shows the result.

The probability and the risk band answer slightly different questions. **Churn: Yes/No** is the model's selected class. **Probability** is its estimated chance of churn. The app labels probability below 30% as **Low**, from 30% to below 60% as **Medium**, and 60% or higher as **High**. These bands are simple labels for reviewing the probability; they are not separate model predictions.

## Why these tools and choices were used

- **React** builds the screens and keeps track of whether the user is filling in a form, viewing a result, or looking at a CSV table. This lets the app change screens without reloading the whole page.
- **HTML and CSS** provide the form controls and the visual design. CSS also handles smaller screens and the dark mode colors.
- **FastAPI** provides a small Python web API. It receives one customer's details or a CSV upload and sends the results back to the React app.
- **scikit-learn** is used because the trained model and its data-preparation steps are saved in a format Python can load and use for predictions.
- **Logistic Regression** is the model used by the app. It was trained as a class-balanced model, which gives the churn class more consideration when the training data has fewer churn examples. It returns a probability used to report the churn risk.
- **Decision Tree** is included in the analysis notebook as a comparison model. The notebook compares models with accuracy, precision, recall, F1-score, and confusion matrices. Keeping the experiment in the notebook makes the model comparison separate from the app that serves predictions.
- **CSV batch prediction** avoids entering many customers one by one. The output only shows the few values needed to review the prediction, while selecting a customer opens the explanation and retention suggestions.
- **Google Gemini API** generates a plain-language explanation and possible follow-up steps for an individual result. It is called only when a user opens a single-customer result or selects a customer from the CSV table. The backend recomputes the model prediction before requesting the explanation, and the API key stays on the backend. Gender and senior-citizen status are not sent to Gemini. Other account, service, and billing details are sent to Gemini for this feature; free-tier prompts may be used to improve Google products. Use sample data only and review AI output for accuracy. A free tier is available within its limits; when the limit is reached, the app displays an error and does not switch to paid use.

## Project parts

- `frontend/` contains the React website.
- `backend/` contains the FastAPI API, its Python packages list, and `churn_model.pkl`, the saved trained model.
- `notebooks/` contains the analysis notebook used to explore and compare models.

## Run the project on Windows

Start the backend in one Command Prompt window:

```bat
cd /d "C:\path\to\churn2\backend"
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Start the website in a second Command Prompt window:

```bat
cd /d "C:\path\to\churn2\frontend"
npm install
npm run dev
```

Open the local website address shown in the frontend window, usually `http://localhost:5173`. Both windows need to stay open while using the app.

## Important notes

- The model expects the same kinds of fields and category values used in its training data. Use the CSV header template from the upload screen.
- Predictions and risk bands are estimates. A business should review the customer's situation before deciding what to do.
- The demo does not save customer records or prediction history. Do not treat it as a secured customer-data system.
- To enable AI explanations locally, set `GEMINI_API_KEY` in the backend process environment before starting FastAPI. Optionally set `GEMINI_MODEL`; it defaults to `gemini-3.1-flash-lite`. Never put the API key in `frontend/.env` or commit it to GitHub.
- On Render, add `GEMINI_API_KEY` to the backend Web Service's Environment Variables, then redeploy that service. Create a Free Tier key in Google AI Studio and leave billing disabled to avoid paid use. Free request quotas apply.
