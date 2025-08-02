import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import gurobipy as gp
from gurobipy import GRB
from math import sqrt

# Streamlit app configuration
st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
        .main { padding: 20px; }
        .stButton > button { background-color: #4CAF50; color: white; border-radius: 5px; }
        .stSlider > div > div > div > div { background-color: #4CAF50; }
        .st-expander { border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px; }
        .stDataFrame { margin-bottom: 20px; }
        h1 { color: #062025; }
        h2 { color: #e74c3c; } /* Vibrant red for headers */
        h3 { color: #2980b9; } /* Vibrant blue for subheaders */
        .sidebar .sidebar-content { background-color: #f8f9fa; }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("📈 Portfolio Optimization Dashboard")

# Default stock list
default_stocks = ['MSFT', 'NVDA', 'AAPL', 'AMZN', 'GOOG', 'GOOGL', 'META', 'AVGO',
                  'BRK-B', 'TSLA', 'WMT', 'JPM', 'V', 'LLY', 'MA', 'NFLX', 'ORCL',
                  'COST', 'XOM', 'PG', 'JNJ', 'HD', 'BAC', 'ABBV', 'KO', 'PLTR', 'PM',
                  'TMUS', 'UNH', 'GE', 'CRM', 'CSCO', 'IBM', 'WFC', 'CVX', 'ABT', 'LIN',
                  'MCD', 'INTU', 'NOW', 'AXP', 'MS', 'DIS', 'T', 'ISRG', 'ACN', 'MRK',
                  'GS', 'AMD', 'RTX']

# 1. Stock Selection and Time Period Selection
with st.expander("🔍 Stock and Time Duration Selection", expanded=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_stocks = st.multiselect(
            "Select at least 2 stocks:",
            options=default_stocks,
            default=['GOOGL', 'META'],
            help="Choose stocks to include in the portfolio analysis."
        )
    with col2:
        time_period = st.selectbox("Select Time Period", ["1y", "2y", "5y"], index=1)

    if len(selected_stocks) < 2:
        st.warning("Please select at least two stocks to proceed.")
    else:
        st.success(f"You selected: {', '.join(selected_stocks)}")

# Fetch stock data
@st.cache_data
def fetch_stock_data(stocks, period):
    data = yf.download(stocks, period=period, auto_adjust=True)
    return data

if len(selected_stocks) >= 2:
    data = fetch_stock_data(selected_stocks, time_period)

    # Compute financial metrics
    closes = np.transpose(np.array(data.Close))
    absdiff = np.diff(closes)
    reldiff = np.divide(absdiff, closes[:, :-1])
    delta = np.mean(reldiff, axis=1)
    sigma = np.cov(reldiff)
    std = np.std(reldiff, axis=1)

    # Update sidebar with parameters
    with st.sidebar:
        st.header("Portfolio Parameters")
        st.write(f"**Selected Stocks**: {', '.join(selected_stocks)}")
        st.write(f"**Time Period**: {time_period}")
        st.write(f"**Number of Stocks**: {len(selected_stocks)}")
        st.write(f"**Trading Days**: {len(data)}")

    # 2. Historical Data Summary
    with st.expander("📊 Historical Data Summary", expanded=True):
        st.subheader("Closing Prices Statistics")
        stats_df = data.Close.describe().transpose()
        st.dataframe(stats_df, use_container_width=True)

        st.subheader("Historical Stock Prices")
        fig = px.line(
            data.Close,
            title="Stock Closing Prices Over Time",
            labels={"value": "Price (USD)", "index": "Date"},
            template="plotly_white"
        )
        fig.update_traces(line=dict(width=2))
        fig.update_layout(
            legend_title_text="Stocks",
            title_x=0.5,
            hovermode="x unified",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3. Financial Metrics
    with st.expander("📋 Financial Metrics", expanded=False):
        st.subheader("Expected Returns and Volatility")
        metrics_df = pd.DataFrame({
            'Stock': selected_stocks,
            'Expected Return (%)': delta * 100,
            'Volatility (%)': std * 100
        })
        st.dataframe(metrics_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Covariance Matrix Heatmap")
            fig = go.Figure(data=go.Heatmap(
                z=sigma,
                x=selected_stocks,
                y=selected_stocks,
                colorscale="RdBu",
                colorbar=dict(title="Covariance"),
                hoverongaps=False
            ))
            fig.update_layout(
                title="Covariance Matrix of Stock Returns",
                title_x=0.5,
                template="plotly_white",
                xaxis_title="Stocks",
                yaxis_title="Stocks"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Returns vs. Volatility")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=selected_stocks,
                y=delta * 100,
                name="Expected Return (%)",
                marker_color="blue"
            ))
            fig.add_trace(go.Bar(
                x=selected_stocks,
                y=std * 100,
                name="Volatility (%)",
                marker_color="orange"
            ))
            fig.update_layout(
                barmode="group",
                title="Expected Returns vs. Volatility",
                title_x=0.5,
                xaxis_title="Stocks",
                yaxis_title="Percentage (%)",
                template="plotly_white",
                legend_title_text="Metrics"
            )
            st.plotly_chart(fig, use_container_width=True)

    # 4. Minimum Risk Portfolio
    with st.expander("🛡️ Minimum Risk Portfolio", expanded=False):
        try:
            # Create and optimize model
            m = gp.Model('portfolio')
            x = m.addMVar(len(selected_stocks), lb=0)
            portfolio_risk = x @ sigma @ x
            m.setObjective(portfolio_risk, GRB.MINIMIZE)
            m.addConstr(x.sum() == 1, 'budget')
            m.optimize()
            minrisk_volatility = sqrt(m.ObjVal)
            minrisk_return = delta @ x.X

            # Update sidebar with minimum risk portfolio metrics
            with st.sidebar:
                st.write(f"**Min Risk Return**: {minrisk_return * 100:.2f}%")
                st.write(f"**Min Risk Volatility**: {minrisk_volatility * 100:.2f}%")

            # Portfolio details
            st.subheader("Portfolio Details")
            portfolio_df = pd.DataFrame({
                'Stock': selected_stocks,
                'Weight (%)': x.X * 100,
                'Expected Return (%)': delta * 100,
                'Volatility (%)': std * 100
            })
            portfolio_df.loc[len(portfolio_df)] = ['Total', x.X.sum() * 100, minrisk_return * 100, minrisk_volatility * 100]
            st.dataframe(portfolio_df, use_container_width=True)

            # Pie chart
            st.subheader("Portfolio Allocation")
            weights = x.X
            labels = [stock if weight > 0.01 else '' for stock, weight in zip(selected_stocks, weights)]
            fig = px.pie(
                values=weights * 100,
                names=labels,
                title="Minimum Risk Portfolio Weights",
                template="plotly_white"
            )
            fig.update_traces(textinfo='percent+label', pull=[0.1 if w > 0.1 else 0 for w in weights])
            fig.update_layout(title_x=0.5, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        except gp.GurobiError:
            st.error("Gurobi license required. Consider using an open-source solver like cvxpy.")

    # 5. Efficient Frontier
    with st.expander("📉 Efficient Frontier", expanded=False):
        # Compute efficient frontier
        frontier = np.empty((2, 0))
        portfolio_return = delta @ x
        target = m.addConstr(portfolio_return == minrisk_return, 'target')
        for r in np.linspace(delta.min(), delta.max(), 25):
            target.rhs = r
            m.optimize()
            frontier = np.append(frontier, [[sqrt(m.ObjVal)], [r]], axis=1)

        # Efficient frontier plot
        st.subheader("Efficient Frontier Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frontier[0] * 100,
            y=frontier[1] * 100,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="DarkGreen", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=std * 100,
            y=delta * 100,
            mode="markers+text",
            name="Individual Stocks",
            marker=dict(size=10, color="Blue"),
            text=selected_stocks,
            textposition="top center"
        ))
        fig.add_trace(go.Scatter(
            x=[minrisk_volatility * 100],
            y=[minrisk_return * 100],
            mode="markers+text",
            name="Minimum Risk Portfolio",
            marker=dict(size=12, color="DarkGreen", symbol="star"),
            text=["Minimum Risk Portfolio"],
            textposition="bottom center"
        ))
        fig.update_layout(
            title="Efficient Frontier",
            title_x=0.5,
            xaxis_title="Volatility (%)",
            yaxis_title="Expected Return (%)",
            template="plotly_white",
            showlegend=True,
            xaxis=dict(range=[frontier[0].min() * 100 * 0.7, frontier[0].max() * 100 * 1.3]),
            yaxis=dict(range=[delta.min() * 100 * 1.2, delta.max() * 100 * 1.2])
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interactive risk-return slider
        st.subheader("Explore Portfolio by Target Return")
        target_return = st.slider(
            "Select Target Return (%)",
            float(delta.min() * 100),
            float(delta.max() * 100),
            float(minrisk_return * 100),
            help="Adjust to explore portfolios with different expected returns."
        )
        target.rhs = target_return / 100
        m.optimize()
        target_volatility = sqrt(m.ObjVal)
        target_weights = x.X
        st.write(f"**Portfolio Volatility**: {target_volatility * 100:.2f}%")
        st.write(f"**Portfolio Expected Return**: {target_return:.2f}%")
        target_df = pd.DataFrame({
            'Stock': selected_stocks,
            'Weight (%)': target_weights * 100
        })
        st.dataframe(target_df, use_container_width=True)

    # 6. Risk Tolerance Adjustment
    with st.expander("⚖️ Risk Tolerance Adjustment", expanded=False):
        risk_tolerance = st.slider(
            "Risk Tolerance (Higher value allows more risk)",
            0.0,
            1.0,
            0.5,
            help="Adjust risk tolerance (currently linked to target return slider)."
        )
        st.info("Risk tolerance is implemented via the target return slider for simplicity.")

    # 7. Download Results
    with st.expander("💾 Download Results", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Download Portfolio Weights",
                portfolio_df.to_csv(index=False),
                "portfolio_weights.csv",
                help="Download the portfolio weights as a CSV file."
            )
        with col2:
            st.download_button(
                "Download Efficient Frontier",
                pd.DataFrame({'Volatility (%)': frontier[0] * 100, 'Expected Return (%)': frontier[1] * 100}).to_csv(index=False),
                "efficient_frontier.csv",
                help="Download the efficient frontier data as a CSV file."
            )
        with col3:
            st.download_button(
                "Download Stock Data",
                data.Close.to_csv(),
                "stock_data.csv",
                help="Download the stock closing prices as a CSV file."
            )