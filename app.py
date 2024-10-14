from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

app = Flask(__name__)

def predict_stock(ticker, start_date, end_date):
    # Fetch historical data based on provided dates
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date']).map(pd.Timestamp.timestamp)

    # Prepare the data for modeling
    X = data[['Date']]
    y = data['Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    data['Predicted'] = model.predict(X)

    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        data = predict_stock(ticker, start_date, end_date)

        # Create the stock price graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Predicted'], mode='lines', name='Predicted'))

        fig.update_layout(title=f'Stock Price Prediction for {ticker}', xaxis_title='Date', yaxis_title='Price (USD)')
        graph = fig.to_html(full_html=False)

        return render_template('index.html', graph=graph)

    return render_template('index.html', graph=None)

if __name__ == '__main__':
    app.run(debug=True)
