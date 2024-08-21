# Data analysis code for weather forecasting
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
FIGSIZE = (12, 6)
ROLLING_WINDOW = 7

def load_data(file_name: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_name)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return None

def calculate_daily_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily statistics."""
    return data.resample('D').agg(['mean', 'std', 'min', 'max'])

def plot_variable(data: pd.DataFrame, variable: str, title: str, ylabel: str) -> None:
    """Plot a variable."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.lineplot(x=data.index, y=data[variable]['mean'], label=variable, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    plt.show()
    plt.close()

def plot_correlation_matrix(data: pd.DataFrame) -> None:
    """Plot the correlation matrix."""
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, ax=ax)
    ax.set_title('Correlation Matrix')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Variables')
    plt.show()
    plt.close()

def plot_scatterplot(data: pd.DataFrame, x: str, y: str) -> None:
    """Plot a scatterplot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=data[x], y=data[y], ax=ax)
    ax.set_title(f'{x} vs. {y}')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.show()
    plt.close()

def plot_distribution(data: pd.DataFrame, variable: str) -> None:
    """Plot a distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(data[variable], kde=False, ax=ax)
    ax.set_title(f'{variable} Distribution')
    ax.set_xlabel(variable)
    ax.set_ylabel('Frequency')
    plt.show()
    plt.close()

def plot_rolling_stats(data: pd.DataFrame, variable: str) -> None:
    """Plot rolling mean and standard deviation."""
    rolling_mean = data[variable].rolling(window=ROLLING_WINDOW).mean()
    rolling_std = data[variable].rolling(window=ROLLING_WINDOW).std()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.lineplot(x=data.index, y=data[variable], label=variable, ax=ax)
    sns.lineplot(x=data.index, y=rolling_mean, label='Rolling Mean', ax=ax)
    sns.lineplot(x=data.index, y=rolling_std, label='Rolling Std Dev', ax=ax)
    ax.set_title(f'Rolling Mean and Std Dev of {variable}')
    ax.set_xlabel('Date')
    ax.set_ylabel(variable)
    ax.legend()
    plt.show()
    plt.close()

def main() -> None:
    data = load_data('weather_data.csv')
    if data is not None:
        daily_stats = calculate_daily_stats(data)
        plot_variable(daily_stats, 'temperature', 'Daily Temperature Means', 'Temperature (°C)')
        plot_variable(daily_stats, 'humidity', 'Daily Humidity Means', 'Humidity (%)')
        plot_variable(daily_stats, 'wind_speed', 'Daily Wind Speed Means', 'Wind Speed (km/h)')
        plot_correlation_matrix(data)
        plot_scatterplot(data, 'temperature', 'humidity')
        plot_distribution(data, 'wind_speed')
        plot_rolling_stats(data, 'temperature')

if __name__ == '__main__':
    main()
#کد تجزیه داده های برای پیش بینی آب و هوا