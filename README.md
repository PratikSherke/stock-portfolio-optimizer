# stock-portfolio-optimizer
📈 Stock Price Prediction Dashboard











The Stock Price Prediction Dashboard is a powerful web application built with Streamlit that enables users to predict stock prices for the next 1–30 days using advanced machine learning models. It combines deep learning (GRU and LSTM) with traditional machine learning algorithms (SVM, Random Forest, XGBoost, and LightGBM) to provide accurate forecasts for a curated list of top-performing stocks.

🚀 Features





Future Price Predictions: Forecasts stock prices for the next 1–30 days based on user-selected prediction horizons.



Multi-Model Predictions: Utilizes GRU, LSTM, SVM, Random Forest, XGBoost, and LightGBM for robust stock price forecasting.



Interactive Dashboard: User-friendly Streamlit interface with customizable parameters for stock selection, lookback days (30–120), and prediction days.



Real-Time Data: Fetches live stock data via the yfinance library for up-to-date market insights.



Interactive Visualizations: Plotly-powered charts displaying historical prices and future predictions with a modern design.



Company Insights: Displays company details, including market capitalization, current price, price change, and business summaries.



Performance Metrics: Evaluates models using MSE, RMSE, MAE, and R², presented in styled tables.



Hyperparameter Tuning: Optimizes models with GridSearchCV for SVM, Random Forest, XGBoost, and LightGBM, and manual tuning for LSTM.



Responsive Design: Custom CSS ensures a visually appealing, responsive UI for desktop and mobile.

🛠️ Installation

Prerequisites





Python 3.8 or higher



pip



Modern web browser (e.g., Chrome, Firefox)

Steps





Clone the Repository:

git clone https://github.com/your-username/stock-price-prediction-dashboard.git
cd stock-price-prediction-dashboard



Create a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Dependencies:

pip install -r requirements.txt



Run the Application:

streamlit run app.py

Requirements

Key dependencies (listed in requirements.txt):





streamlit



numpy



pandas



yfinance



plotly



scikit-learn



tensorflow



xgboost



lightgbm

Install manually with:

pip install streamlit numpy pandas yfinance plotly scikit-learn tensorflow xgboost lightgbm

📊 Usage





Select Stocks: Choose from 50 top companies (e.g., MSFT, AAPL, NVDA) in the sidebar.



Set Parameters: Adjust lookback days (30–120) and prediction days (1–30) using sliders.



Run Analysis: Click "Run Analysis" to fetch data, train models, and generate predictions.



View Results: Explore company info, recent prices, prediction charts, model metrics, predicted prices, and hyperparameters.

🧠 Models and Algorithms





GRU (Gated Recurrent Unit): Lightweight RNN for capturing temporal patterns.



LSTM (Long Short-Term Memory): Robust RNN with tuned units, activations, and learning rates.



SVM (Support Vector Machine): Optimized via GridSearchCV for regression.



Random Forest: Ensemble method with tuned hyperparameters.



XGBoost: High-performance gradient boosting algorithm.



LightGBM: Efficient gradient boosting for large datasets.

Models are trained on historical data, predicting prices for the next 1–30 days. Performance is evaluated with MSE, RMSE, MAE, and R².

📈 Visualizations

Interactive Plotly charts show:





Historical prices for the last 30 days.



Predicted prices for the user-specified forecast period.



A connecting line between actual and predicted prices.

Charts are styled with custom CSS and Plotly's white template.

📝 Data Sources





Stock Data: Sourced from Yahoo Finance via yfinance.



Company Information: Includes market cap, current price, and business summaries from Yahoo Finance.

🔧 Customization





Modify the stock_options list to include different stocks.



Adjust hyperparameters (RNN_UNITS, ACTIVATIONS, LEARNING_RATES) for deep learning models.



Tune TEST_SPLIT, BATCH_SIZE, and EPOCHS for training.



Customize CSS in the st.markdown section for UI enhancements.

📚 Future Enhancements





Add models like Prophet or ARIMA.



Implement real-time price streaming.



Incorporate sentiment analysis from financial news or X posts.



Enable export of predictions and metrics as CSV or PDF.



Optimize training with GPU acceleration.

🤝 Contributing





Fork the repository.



Create a feature branch (git checkout -b feature/your-feature).



Commit changes (git commit -m 'Add your feature').



Push to the branch (git push origin feature/your-feature).



Open a pull request.

Ensure code follows PEP 8 and includes tests.

🙏 Acknowledgments





Streamlit for the web app framework.



Yahoo Finance for stock data.



TensorFlow and scikit-learn for ML tools.



Plotly for visualizations.

📬 Contact

Reach out via GitHub Issues or on X.

