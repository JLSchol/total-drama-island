import pandas as pd
import matplotlib.pyplot as plt
from process import get_activities_df




def plot_profit_loss(df):
    # List of unique products in the dataset
    products = df['product'].unique()

    # Create a figure
    plt.figure(figsize=(12, 6))

    # Plot Profit and Loss for each product
    for product in products:
        product_data = df[df['product'] == product]
        
        # Plot Profit and Loss for each product
        plt.plot(product_data['timestamp'], product_data['profit_and_loss'], label=product, marker='o')

    # Labels and Formatting
    plt.xlabel("Timestamp")
    plt.ylabel("Profit and Loss")
    plt.title("Profit and Loss for Each Product Over Time")
    plt.legend(title="Product")
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    

def plot_mid_price(df):
    # List of unique products in the dataset
    products = df['product'].unique()

    # Create a figure
    plt.figure(figsize=(12, 6))

    # Plot Mid Price for each product
    for product in products:
        product_data = df[df['product'] == product]
        
        # Plot Mid Price for each product
        plt.plot(product_data['timestamp'], product_data['mid_price'], label=product, marker='o')

    # Labels and Formatting
    plt.xlabel("Timestamp")
    plt.ylabel("Mid Price")
    plt.title("Mid Price for Each Product Over Time")
    plt.legend(title="Product")
    plt.grid(True)

    # Show the plot
    plt.tight_layout()


if __name__ == '__main__':
    # # Load Data
    # csv_file1 = r"C:\Users\jasschol\repositories\personal projects\total-drama-island\logs\tutorial\2503041722_example_upload\processed\2503041722_example_upload_activities.csv"
    # csv_file2 = r"C:\Users\jasschol\repositories\personal projects\total-drama-island\logs\tutorial\2503191612_buy_everything\processed\2503191612_buy_everything_activities.csv"
    # csv_file3 = r"C:\Users\jasschol\repositories\personal projects\total-drama-island\logs\tutorial\2503191644_do_nothing\processed\2503191644_do_nothing_activities.csv"
    # csv_file4 = r"C:\Users\jasschol\repositories\personal projects\total-drama-island\logs\tutorial\250323_ma\processed\250323_ma_activities.csv"
    # for file in [csv_file1,  csv_file2,  csv_file3,  csv_file4]:
    #     df = get_activities_df(file)
    #     plot_profit_loss(df)
    #     # plot_mid_price(df)
    csv_file4 = r"C:\Users\jasschol\repositories\personal projects\total-drama-island\logs\tutorial\250323_ma\processed\250323_ma_activities.csv"
    df = get_activities_df(csv_file4)
    plot_profit_loss(df)
    plt.show()
