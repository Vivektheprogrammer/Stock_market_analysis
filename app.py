# app.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, avg, when, to_date
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# -----------------------------
# Configure Spark temp dir (Windows-safe)
# -----------------------------
os.environ['SPARK_LOCAL_DIRS'] = r"C:\SparkTemp"
os.makedirs(os.environ['SPARK_LOCAL_DIRS'], exist_ok=True)

# -----------------------------
# Initialize Spark
# -----------------------------
spark = SparkSession.builder \
    .appName("StockPricePrediction") \
    .getOrCreate()

# -----------------------------
# Streamlit UI - File Upload
# -----------------------------
st.title("📈 Multi-Ticker Stock Price Predictor with Sector View")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file:
    # Preview file using Pandas
    df_pd = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded CSV:", df_pd.head())

    # -----------------------------
    # Add Sector column
    # -----------------------------
    sector_map = {
        'TCS.NS': 'Tech', 'INFY.NS': 'Tech', 'WIPRO.NS': 'Tech', 'HCLTECH.NS': 'Tech', 'TECHM.NS': 'Tech',
        'BAJAJHLDNG.NS': 'Auto', 'LT.NS': 'Auto', 'MARUTI.NS': 'Auto', 'TATAMOTORS.NS': 'Auto', 
        'BAJAJ-AUTO.NS': 'Auto', 'EICHERMOT.NS': 'Auto', 'M&M.NS': 'Auto',
        'HDFCBANK.NS': 'Bank', 'ICICIBANK.NS': 'Bank', 'KOTAKBANK.NS': 'Bank', 'SBIN.NS': 'Bank', 'AXISBANK.NS': 'Bank',
        'SUNPHARMA.NS': 'Pharma', 'CIPLA.NS': 'Pharma', 'DIVISLAB.NS': 'Pharma', 'DRREDDY.NS': 'Pharma', 'ABBOTINDIA.NS': 'Pharma',
        'GRSE.NS': 'Defense', 'HAL.NS': 'Defense', 'BEL.NS': 'Defense'
    }
    df_pd['Sector'] = df_pd['Ticker'].map(sector_map)

    # Save to temp location for PySpark
    temp_csv_path = r"C:\SparkTemp\uploaded_stock_data.csv"
    df_pd.to_csv(temp_csv_path, index=False)

    # -----------------------------
    # Load CSV in PySpark
    # -----------------------------
    df = spark.read.csv(temp_csv_path, header=True, inferSchema=True)

    # Clean + convert Date
    df = df.withColumn('Date', to_date(col('Date'), 'yyyy-MM-dd'))
    df = df.dropna(subset=['Date', 'Ticker', 'Close'])

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    w = Window.partitionBy('Ticker').orderBy('Date')
    df = df.withColumn('Prev_Close', lag('Close', 1).over(w))
    df = df.withColumn('Prev_Volume', lag('Volume', 1).over(w))
    df = df.withColumn('MA_5', avg('Close').over(w.rowsBetween(-4, 0)))
    df = df.withColumn('MA_10', avg('Close').over(w.rowsBetween(-9, 0)))
    df = df.withColumn('Return',
                       when(col('Prev_Close') != 0, (col('Close') - col('Prev_Close')) / col('Prev_Close')).otherwise(0))
    df = df.na.drop()

    # -----------------------------
    # Assemble features
    # -----------------------------
    feature_cols = ['Prev_Close', 'Prev_Volume', 'MA_5', 'MA_10', 'Return']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    df = assembler.transform(df)

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # -----------------------------
    # Train Random Forest
    # -----------------------------
    rf = RandomForestRegressor(featuresCol='features', labelCol='Close', numTrees=200, maxDepth=10)
    model = rf.fit(train_df)

    # -----------------------------
    # Predictions
    # -----------------------------
    predictions = model.transform(test_df)
    evaluator_rmse = RegressionEvaluator(labelCol='Close', predictionCol='prediction', metricName='rmse')
    evaluator_r2 = RegressionEvaluator(labelCol='Close', predictionCol='prediction', metricName='r2')
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    st.success(f"Model Performance: RMSE={rmse:.4f}, R²={r2:.4f}")

    # -----------------------------
    # Sector Selection
    # -----------------------------
    sectors = [row['Sector'] for row in df.select('Sector').distinct().collect()]
    selected_sector = st.selectbox("Select Sector", ["All"] + sectors)

    # -----------------------------
    # Ticker Selection
    # -----------------------------
    tickers = [row['Ticker'] for row in df.select('Ticker').distinct().collect()]
    selected_ticker = st.selectbox("Select Ticker to plot", tickers)

    # -----------------------------
    # Filter Data for Selected Ticker + Sector
    # -----------------------------
    if selected_sector != "All":
        pred_pd = predictions.filter((col('Ticker') == selected_ticker) & (col('Sector') == selected_sector)) \
                             .select('Date', 'Ticker', 'Close', 'prediction').toPandas()
    else:
        pred_pd = predictions.filter(col('Ticker') == selected_ticker) \
                             .select('Date', 'Ticker', 'Close', 'prediction').toPandas()

    pred_pd.sort_values('Date', inplace=True)

    # -----------------------------
    # Plot Ticker Actual vs Predicted
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(pred_pd['Date'], pred_pd['Close'], label='Actual')
    plt.plot(pred_pd['Date'], pred_pd['prediction'], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f"{selected_ticker} - Actual vs Predicted")
    plt.legend()
    st.pyplot(plt)

    # -----------------------------
    # Option: Plot all tickers in sector
    # -----------------------------
    if st.checkbox("Plot all tickers in sector"):
        if selected_sector != "All":
            sector_df = predictions.filter(col('Sector') == selected_sector) \
                                   .select('Date', 'Ticker', 'Close').toPandas()
            sector_df['Date'] = pd.to_datetime(sector_df['Date'])
            fig = px.line(sector_df, x='Date', y='Close', color='Ticker',
                          title=f"{selected_sector} Sector - Actual Close Prices")
            st.plotly_chart(fig)
        else:
            st.info("Please select a specific sector to plot multiple tickers.")
