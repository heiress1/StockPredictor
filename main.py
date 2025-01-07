import streamlit as st
from datetime import date
#pandas is a data manipulation library
import pandas as pd
import yfinance as yf 
from prophet import Prophet
from prophet.plot import plot_plotly

#plotly graph is visually prettier than matplotlib
from plotly import graph_objs as go

START = "2015-01-01"

#date.today gets today's date
#strftime formats the date into readable string representations
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

#lets the user type in a stock ticker
ticker_input = st.text_input("Enter a stock ticker (e.g. to track the Canadian S&P5OO: VFV.TO)", "VFV.TO")


# stocks = ("AAPL", "GOOG", "MSFT", "GME")
#selectbox is a dropdown menu
# selected_stock = st.selectbox("Select dataset for prediction", stocks)

#creates a slider to select the number of years to predict
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# If the user has NOT entered any ticker, show a warning and stop
if not ticker_input.strip():
    st.warning("Please enter a ticker to proceed.")
    st.stop()

#when st.cache is applyed to a function, it will cache the functions output
#this means that when the function is called with the same argument,
#streamlit will return the cached(saved) result instead of re-executing the function 
#cache essentially means saved
@st.cache_data
def load_data(ticker: str) -> pd.DataFrame:
    #downloads the stock data from the START date to TODAY
    #will return the data in a pandas dataframe
    data = yf.download(ticker, START, TODAY)

    #puts the date in the first column of the pandas dataframe
    data.reset_index(inplace=True)

     # yfinance returns a multi-index dataframe
     # making so you cannot access the columns directly with data['Open'] '
     #for example the mutli-index dataframe will have columns like ("Open","AAPL")
     #so since we are downloading one ticker at a time, we can flatten the columns so you can access the columns directly with data['Open']
    #this condition flattens the columns if the columns are a multi-index

    if isinstance(data.columns, pd.MultiIndex):
        #turns ("Open","AAPL") into "Open AAPL"
        data.columns = [' '.join(col).strip() for col in data.columns]        
        
    
    # We rename the columns to remove the ticker name so we can access the columns directly with data['Open']
    # For example, rename "Open AAPL" => "Open"
    rename_dict = {}
    for col in data.columns:
        if "Open" in col:       rename_dict[col] = "Open"
        elif "High" in col:     rename_dict[col] = "High"
        elif "Low" in col:      rename_dict[col] = "Low"
        elif "Close" in col:    rename_dict[col] = "Close"
        elif "Adj Close" in col:rename_dict[col] = "Adj Close"
        elif "Volume" in col:   rename_dict[col] = "Volume"
    data.rename(columns=rename_dict, inplace=True)

    if data.columns.duplicated().any():
        # Return an empty DataFrame as a signal that this ticker is invalid/ambiguous
        return pd.DataFrame()
    
    return data

#while the data is loading, the text "Load data..." will be displayed
data_load_state = st.text("Load data...")


#data is a pandas dataframe
data = load_data(ticker_input)

#once the data is loaded, the text will change to "Data loaded!"
data_load_state = st.text("Data Loaded!")

#if ticker is not valid, sends an error message and stops the app
if data.empty:
    st.error(f"Ticker '{ticker_input}' not found or returned ambiguous data. Please try a different symbol.")
    st.stop()

st.subheader('Raw data')

#displays the raw data
st.write(data.tail())

def plot_raw_data():
    # go.Figure creates a new figure object with plotly
    fig = go.Figure()

    # Adds a scatter plot to the figure
    # pandas dataframe is similar to a dictionary where you can access the columns using the key
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    # Display the figure in the Streamlit app
    st.plotly_chart(fig)

# Call the function to plot the raw data
st.write("You can use the slider below the graph to focus on a specific time period")
plot_raw_data()


#predicts the stock price using the prophet library

df_train = data[['Date', 'Close']]

#facebook prophet requres the columns to be named in a specific way
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

#create a prophet model
model = Prophet()

#the fit method in prophet trains the model using the df_train dataframe
model.fit(df_train) 

#creates a new dataframe what extends from the last date in the training data to a future date
#periods=period specifies the number of future time periods in new dataframe
future = model.make_future_dataframe(periods=period)

#predicts the future stock prices 
#using the predict method of prophet model to generate forecasts for the specified future dates (periods)
#future contains dates for which you want to make predictions
forecast = model.predict(future)

#displays the forecast data in a table
st.subheader('Predicted data')
st.write(forecast.tail())

st.subheader("Predicted Graph")
st.write("Black dots represent the actual data points")
st.write("Blue line represents the predicted stock price")
st.write("You can use the slider below the graph to focus on a specific time period")

#display forecast data as a graph
figure1 = plot_plotly(model, forecast)

st.plotly_chart(figure1)

st.subheader("Forecast Components to Show Trends on a Weekly and Yearly Basis")

figure2 = model.plot_components(forecast)
st.write(figure2)

