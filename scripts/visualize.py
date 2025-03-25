import pandas as pd
import matplotlib.pyplot as plt
from process import get_activities_df
import os



def plot_profit_loss(df, save_path):
    """Plots profit and loss for each product and saves the figure."""
    products = df['product'].unique()

    plt.figure(figsize=(12, 6))

    for product in products:
        product_data = df[df['product'] == product]
        plt.plot(product_data['timestamp'], product_data['profit_and_loss'], label=product, marker='o')

    plt.xlabel("Timestamp")
    plt.ylabel("Profit and Loss")
    plt.title("Profit and Loss for Each Product Over Time")
    plt.legend(title="Product")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    

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
    base_path = r"C:\Users\jasschol\repositories\personal projects\total-drama-island\logs\tutorial"
    
    directories = ["250325_dc"]  # Add more directories as needed

    for directory in directories:
        csv_file = os.path.join(base_path, directory, "processed", f"{directory}_activities.csv")
        plot_path = os.path.join(base_path, directory, "plots", "profit_and_loss.png")

        if os.path.exists(csv_file):
            df = get_activities_df(csv_file)
            plot_profit_loss(df, plot_path)
            print(f"Plot saved: {plot_path}")
        else:
            print(f"File not found: {csv_file}")
    plt.show()