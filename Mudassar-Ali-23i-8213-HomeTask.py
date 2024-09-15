import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# data = pd.read_csv(r'E:\FAST-MS-2023-2025\3RD-SEMESTER\1-Machine Learning\Mudassar Ali-8213-Home-Task-No-1\NumpyRegCSV_Data.csv')
data = pd.read_csv('NumpyRegCSV_Data.csv')

print(data.describe())
print(data.info())

data = data.dropna()

X = data[['Duration', 'Pulse', 'Maxpulse']]
y = data['Calories']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def run_regression(X_train, X_test, y_train, y_test, split_ratio):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"\n--- Results for {split_ratio}% ---")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")
    print(f"MAE: {mae}")
    
    plt.scatter(y_test, predictions, color='blue')
    plt.plot(y_test, y_test, color='red')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted ({split_ratio}% split)')
    plt.grid(True)
    plt.show()

for split in [0.2, 0.4, 0.8]:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=split, random_state=42)
    run_regression(X_train, X_test, y_train, y_test, int(split*100))
