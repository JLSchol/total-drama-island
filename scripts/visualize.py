import pandas as pd
import matplotlib.pyplot as plt
from process import get_activities_df
import os



def plot_profit_loss(df, save_path):
    """Plots profit and loss for each product and saves the figure."""
    products = df['product'].unique()

    plt.figure(figsize=(12, 6))
    # Dictionary to store final profit/loss per product
    final_profits = {}
    for product in products:
        product_data = df[df['product'] == product]
        plt.plot(product_data['timestamp'], product_data['profit_and_loss'], label=product, marker='o')
        # Store the final profit/loss for each product
        final_profits[product] = product_data['profit_and_loss'].iloc[-1]
    # Calculate total final profit/loss
    total_profit = df[df['timestamp'] == df['timestamp'].iloc[-1]]['profit_and_loss'].sum()
    
    final_profits["TOTAL"] = total_profit 

    plt.xlabel("Timestamp")
    plt.ylabel("Profit and Loss")
    plt.title("Profit and Loss for Each Product Over Time")
    plt.legend(title="Product")
    plt.grid(True)

    # Display final profits on the plot
    text_y_position = min(df['profit_and_loss']) - 50  # Adjust position to avoid overlap
    text = "\n".join([f"{product}: {profit:.2f} seashells" for product, profit in final_profits.items()])
    # text += f"\nTotal: {total_profit:.2f} seashells"
    plt.text(df['timestamp'].iloc[-1000], text_y_position, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    return final_profits
    

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
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the base path relative to the script's location
    base_path = os.path.join(script_dir, '..', 'logs', 'tutorial')

    # what to visualize and load
    directories = ["250323_ma_second_best", "250325_dc"]  # Add more directories as needed

    for directory in directories:
        csv_file = os.path.join(base_path, directory, "processed", f"{directory}_activities.csv")
        plot_path = os.path.join(base_path, directory, "plots", "profit_and_loss.png")

        if os.path.exists(csv_file):
            df = get_activities_df(csv_file)
            penl = plot_profit_loss(df, plot_path)
            print(f"Plot saved: {plot_path}")
            print("\n")
            print(f"{directory}: {penl=}")
            print("---\n")
        else:
            print(f"File not found: {csv_file}")
    plt.show()