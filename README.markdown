# 📈 Portfolio Optimization Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red) ![Gurobi](https://img.shields.io/badge/Gurobi-9.0%2B-orange)

The **Portfolio Optimization Dashboard** is a sophisticated web application built with Streamlit that enables users to construct optimized investment portfolios using modern portfolio theory. Leveraging the Gurobi optimization solver, it analyzes historical stock data to minimize risk while targeting desired returns, offering an intuitive interface for financial analysis and portfolio construction.

## 🚀 Features

- **Portfolio Optimization**: Minimizes portfolio risk for a given expected return using Gurobi's quadratic optimization, computing the efficient frontier.
- **Interactive Stock Selection**: Choose from a curated list of 50 top-performing stocks (e.g., MSFT, AAPL, NVDA) with customizable time periods (1, 2, or 5 years).
- **Real-Time Data**: Fetches historical stock data via the `yfinance` library for accurate financial metrics.
- **Comprehensive Visualizations**: Interactive Plotly charts, including historical price trends, covariance heatmaps, returns vs. volatility bar charts, efficient frontier plots, and portfolio allocation pie charts.
- **Financial Metrics**: Displays expected returns, volatility, and covariance matrices for selected stocks.
- **Dynamic Risk Adjustment**: Explore portfolios along the efficient frontier with a target return slider and view corresponding weights and volatility.
- **Data Export**: Download portfolio weights, efficient frontier data, and stock prices as CSV files for further analysis.
- **Responsive Design**: Custom CSS ensures a visually appealing, user-friendly interface optimized for desktop and mobile.

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip
- Gurobi license (required for optimization; academic licenses available)
- Modern web browser (e.g., Chrome, Firefox)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/portfolio-optimization-dashboard.git
   cd portfolio-optimization-dashboard
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

   This launches the dashboard in your default web browser.

### Requirements
Key dependencies (listed in `requirements.txt`):
- `streamlit`
- `yfinance`
- `numpy`
- `pandas`
- `plotly`
- `gurobipy`
- `seaborn`

Install manually with:
```bash
pip install streamlit yfinance numpy pandas plotly gurobipy seaborn
```

**Note**: A valid Gurobi license is required. If unavailable, consider using an open-source solver like CVXPY as an alternative.

## 📊 Usage

1. **Select Stocks and Time Period**: Choose at least two stocks from the list and a time period (1, 2, or 5 years) in the "Stock and Time Duration Selection" section.
2. **View Historical Data**: Explore closing price statistics and interactive price trend charts in the "Historical Data Summary" section.
3. **Analyze Financial Metrics**: Review expected returns, volatility, covariance heatmaps, and returns vs. volatility comparisons.
4. **Explore Minimum Risk Portfolio**: View the optimized portfolio weights, expected return, volatility, and allocation pie chart.
5. **Interact with Efficient Frontier**: Adjust the target return slider to explore portfolios along the efficient frontier and view corresponding weights and volatility.
6. **Download Results**: Export portfolio weights, efficient frontier data, and stock prices as CSV files.

## 🧠 Optimization and Analysis

- **Portfolio Optimization**: Uses Gurobi to solve a quadratic optimization problem, minimizing portfolio risk (variance) subject to a budget constraint (weights sum to 1) and optional target return constraints.
- **Efficient Frontier**: Computes portfolios with minimum risk for a range of expected returns, visualized as a curve plotting volatility against return.
- **Financial Metrics**: Calculates expected returns (mean daily returns), volatility (standard deviation of returns), and covariance matrix of relative price changes.

## 📈 Visualizations

Interactive Plotly charts include:
- **Historical Prices**: Line charts of stock closing prices over the selected period.
- **Covariance Heatmap**: Visualizes correlations between stock returns.
- **Returns vs. Volatility**: Bar chart comparing expected returns and volatility for each stock.
- **Efficient Frontier**: Scatter plot showing the trade-off between risk and return, with individual stocks and the minimum risk portfolio highlighted.
- **Portfolio Allocation**: Pie chart displaying optimized portfolio weights.

## 📝 Data Sources

- **Stock Data**: Historical closing prices sourced from Yahoo Finance via `yfinance`.

## 🔧 Customization

- Modify the `default_stocks` list to include different stocks.
- Adjust the time period options in the `selectbox` (e.g., add "3m" or "10y").
- Enhance CSS in the `st.markdown` section for a personalized UI.
- Replace Gurobi with an open-source solver like CVXPY for license-free operation.

## 📚 Future Enhancements

- Integrate additional solvers (e.g., CVXPY) for broader accessibility.
- Add support for constraints like sector diversification or maximum weight limits.
- Incorporate real-time stock price streaming.
- Include risk metrics like Sharpe Ratio or Value at Risk (VaR).
- Enable sentiment analysis from financial news or X posts for enhanced insights.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web app framework.
- [Yahoo Finance](https://finance.yahoo.com/) for stock data.
- [Gurobi](https://www.gurobi.com/) for optimization capabilities.
- [Plotly](https://plotly.com/) for interactive visualizations.


Optimize your investments! 📈
