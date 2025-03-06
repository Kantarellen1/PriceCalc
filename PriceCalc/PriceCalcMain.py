import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def filter_data_by_area(data, area):
    filtered_data = data[data['Area'] == area]
    return filtered_data

def filter_data_by_postal_code(data, postal_code):
    filtered_data = data[data['PostalCode'] == postal_code]
    return filtered_data

def filter_data_by_year(data, years):
    current_year = pd.Timestamp.now().year
    start_year = current_year - years
    filtered_data = data[data['Year'] >= start_year]
    return filtered_data

def calculate_median_price(data):
    prices = data['Price'].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_prices = scaler.fit_transform(prices)
    
    scaled_prices_tensor = torch.tensor(scaled_prices, dtype=torch.float32)
    
    median_price_tensor = torch.median(scaled_prices_tensor)
    
    median_price = scaler.inverse_transform(median_price_tensor.unsqueeze(0).numpy().reshape(1, -1))[0][0]
    return median_price

def is_good_deal(current_price, median_price):
    return current_price < median_price

def main():
    file_path = 'C:/Users/Kalle/Documents/GitHub/PriceCalc/housing_data.csv' 
    area = 'Copenhagen'  
    postal_code = '1234'  
    current_price = 300000  

    data = load_data(file_path)
    data = filter_data_by_area(data, area)
    data = filter_data_by_postal_code(data, postal_code)
    data = filter_data_by_year(data, 5)
    
    median_price = calculate_median_price(data)
    good_deal = is_good_deal(current_price, median_price)
    
    print(f"The median price in {area} with postal code {postal_code} over the last 5 years is: ${median_price}")
    if good_deal:
        print("The current price is a good deal.")
    else:
        print("The current price is not a good deal.")

if __name__ == "__main__":
    main()