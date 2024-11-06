### DEVELOPED BY: Nivetha A
### REGISTER NO: 212222230101
### DATE:


# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def arima_model(data, target_variable, order):
    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Fit the ARIMA model on the training data
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    # Forecast for the length of the test data
    forecast = fitted_model.forecast(steps=len(test_data))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='red')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    # Print the RMSE value
    print("Root Mean Squared Error (RMSE):", rmse)

# Load the dataset
data = pd.read_csv('raw_sales.csv')

# Convert 'datesold' to datetime and set as index
data['datesold'] = pd.to_datetime(data['datesold'])
data.set_index('datesold', inplace=True)

# Run the ARIMA model function with the specified target and order
arima_model(data, 'price', order=(5,1,0))
```
### OUTPUT:
![382584145-a8279c2b-cdb7-44d0-9a97-7124e9e857ce](https://github.com/user-attachments/assets/1bf00707-c921-4329-92ff-d960708a70ff)


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
