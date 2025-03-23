# load activities
import csv
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt

def extract_columns_starting_with(name: str, df_columns):
    return sorted([col for col in df_columns if col.startswith(name)])

def get_activities_dict(abs_csv_file_path: str) -> dict:
    activities_dict = {}

    with open(abs_csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')  # CSV uses `;` as delimiter
        headers = next(reader)  # Read the header row

        # Identify indices for bid/ask price and volume
        bid_price_indices = [i for i, h in enumerate(headers) if "bid_price" in h]
        bid_volume_indices = [i for i, h in enumerate(headers) if "bid_volume" in h]
        ask_price_indices = [i for i, h in enumerate(headers) if "ask_price" in h]
        ask_volume_indices = [i for i, h in enumerate(headers) if "ask_volume" in h]
        # Find indices for mid_price and profit_and_loss
        mid_price_index = headers.index("mid_price")
        profit_and_loss_index = headers.index("profit_and_loss")


        for row in reader:
            day, timestamp, product, *values = row

            timestamp = int(timestamp)  # Convert timestamp to integer
            
            # Extract bid and ask data dynamically
            bid_prices = [float(row[i]) if row[i] else None for i in bid_price_indices]
            bid_volumes = [float(row[i]) if row[i] else None for i in bid_volume_indices]
            ask_prices = [float(row[i]) if row[i] else None for i in ask_price_indices]
            ask_volumes = [float(row[i]) if row[i] else None for i in ask_volume_indices]

            # Remove None values
            bids = {bid_prices[i]: bid_volumes[i] for i in range(len(bid_prices)) if bid_prices[i] is not None}
            asks = {ask_prices[i]: ask_volumes[i] for i in range(len(ask_prices)) if ask_prices[i] is not None}

            mid_price = float(values[mid_price_index - 3]) if values[mid_price_index - 3] else None
            profit_and_loss = float(values[profit_and_loss_index - 3]) if values[profit_and_loss_index - 3] else None

            # Insert into dictionary structure
            if timestamp not in activities_dict:
                activities_dict[timestamp] = {}

            activities_dict[timestamp][product] = {
                "bids": bids,
                "asks": asks,
                "mid_price": mid_price,
                "profit_and_loss": profit_and_loss
            }

    return activities_dict

def get_activities_df(abs_csv_file_path: str) -> pd.DataFrame:
    return pd.read_csv(abs_csv_file_path, delimiter=';')  # Use ';' as the delimiter based on your file

def add_weighted_prices(df):
    def weighted_avg_price(price_cols, volume_cols, df):
        total_value = sum(df[price].fillna(0) * df[volume].fillna(0) for price, volume in zip(price_cols, volume_cols))
        total_volume = sum(df[volume].fillna(0) for volume in volume_cols)
        
        # Avoid division by zero, return NaN where total volume is zero
        return total_value / total_volume.where(total_volume != 0, float("nan"))

    # Dynamically extract column names for bid and ask prices/volumes
    bid_prices = extract_columns_starting_with("bid_price",df.columns)
    bid_volumes = extract_columns_starting_with("bid_volume",df.columns)
    ask_prices = extract_columns_starting_with("ask_price",df.columns)
    ask_volumes = extract_columns_starting_with("ask_volume",df.columns)

    # Compute best bid and best ask prices
    df["best_bid_price"] = df[bid_prices].max(axis=1, skipna=True)  # Highest bid price
    df["best_ask_price"] = df[ask_prices].min(axis=1, skipna=True)  # Lowest ask price

    # Compute total bid and ask volumes
    df["total_bid_volume"] = df[bid_volumes].sum(axis=1, min_count=1)  # Sum with NaN handling
    df["total_ask_volume"] = df[ask_volumes].sum(axis=1, min_count=1)

    # Compute weighted average bid/ask prices dynamically
    df["weighted_bid_price"] = weighted_avg_price(bid_prices, bid_volumes, df)
    df["weighted_ask_price"] = weighted_avg_price(ask_prices, ask_volumes, df)

    # Compute weighted mid price
    df["weighted_mid_price"] = (
        df["weighted_bid_price"] * df["total_ask_volume"] + df["weighted_ask_price"] * df["total_bid_volume"]
    ) / (df["total_bid_volume"] + df["total_ask_volume"]).where((df["total_bid_volume"] + df["total_ask_volume"]) != 0, float("nan"))

    # Market structure and liquidity
    # bid ask spread: A tight spread suggests a liquid market, wide spread suggest volatile market -> find quick in and out trades
    df["bid_ask_spread"] = df["best_ask_price"] - df["best_bid_price"]
    # Market Depth (Liquidity Ratio): High liquidity ratio (>1) → More buyers (<1) more sellers) -> trend following
    df["liquidity_ratio"] = df["total_bid_volume"] / (df["total_ask_volume"] + 1e-6)

    return df





def plot_product_data(df):
    """Plots weighted bid, ask, and mid prices, along with total bid and ask volumes per product."""
    
    products = df["product"].unique()

    for product in products:
        product_df = df[df["product"] == product]

        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 8), sharex=True)

        # Subplot 1: Price trends
        axes[0].plot(product_df["timestamp"], product_df["best_bid_price"], label="Best Bid Price", color="blue")
        axes[0].plot(product_df["timestamp"], product_df["best_ask_price"], label="Best Ask Price", color="red")
        axes[0].plot(product_df["timestamp"], product_df["mid_price"], label="Mid Price", color="purple", linestyle="dashed")

        axes[0].set_ylabel("Price")
        axes[0].set_title(f"Best Price Trends for {product}")
        axes[0].legend()
        axes[0].grid(True)

        # Subplot 2: weigthed price trend
        axes[1].plot(product_df["timestamp"], product_df["weighted_bid_price"], label="Weighted Bid Price", color="blue")
        axes[1].plot(product_df["timestamp"], product_df["weighted_ask_price"], label="Weighted Ask Price", color="red")
        axes[1].plot(product_df["timestamp"], product_df["weighted_mid_price"], label="Weighted Mid Price", color="purple", linestyle="dashed")
        axes[1].set_ylabel("Price")
        axes[1].set_title(f"Weighted Price Trends for {product}")
        axes[1].legend()
        axes[1].grid(True)

        # Subplot 3: Volume trends
        axes[2].plot(product_df["timestamp"], product_df["total_bid_volume"], label="Total Bid Volume", color="blue")
        axes[2].plot(product_df["timestamp"], product_df["total_ask_volume"], label="Total Ask Volume", color="red")
        axes[2].set_xlabel("Timestamp")
        axes[2].set_ylabel("Volume")
        axes[2].set_title(f"Volume Trends for {product}")
        axes[2].legend()
        axes[2].grid(True)

        # subplot4:  
        axes[3].plot(product_df["timestamp"], product_df["bid_ask_spread"], label="bid ask spread", color="blue")
        axes[3].set_xlabel("Timestamp")
        axes[3].set_ylabel("spread")
        axes[3].set_title(f"bid ask spread for {product}")
        axes[3].legend()
        axes[3].grid(True)
        # subplot5:  
        axes[4].plot(product_df["timestamp"], product_df["liquidity_ratio"], label="liquidity ratio", color="blue")
        axes[4].set_xlabel("Timestamp")
        axes[4].set_ylabel("ratio")
        axes[4].set_ylim(-1,3)
        axes[4].set_title(f"liquidity ratio for {product}")
        axes[4].legend()
        axes[4].grid(True)

        plt.tight_layout()


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
    plt.ylabel("Seashells earned")
    plt.title("Profit and Loss for Each Product Over Time")
    plt.legend(title="Product")
    plt.grid(True)

    # Show the plot
    plt.tight_layout()

if __name__ == "__main__":
    # activity_csv = r"C:\Users\jasschol\repositories\personal projects\total-drama-island\logs\tutorial\250319_ma\processed\250319_ma_activities.csv"
    # activity_csv = r"C:\Users\jasschol\repositories\personal projects\total-drama-island\logs\tutorial\250320_ma\processed\250320_ma_activities.csv"
    activity_csv = r"C:\Users\jasschol\repositories\personal projects\total-drama-island\logs\tutorial\250323_ma\processed\250323_ma_activities.csv"

    # activities_dict = get_activities_dict(activity_csv)
    df = get_activities_df(activity_csv)
    # print(df.columns[0])

    df = add_weighted_prices(df)  # Ensure weighted prices are calculated
    plot_product_data(df)

    plot_profit_loss(df)
    plt.show()
    # plot_weighted_prices(df)  # Plot weighted bid/ask prices

    # print(df.head())


    # trade_history = get_trade_history()
    # sandbox = get_sandbox()

