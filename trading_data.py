import pandas as pd
import json
from typing import Dict, List, Optional
from mock import get_state


class TradingData:
    def __init__(self, state, position_limits: dict[str,int], max_depth: int=3):
        """
        Initialize TradingData from the state object. If traderData exists, load it into a DataFrame.
        Otherwise, start with an empty DataFrame.
        """
        self.state = state
        self.position_limits = position_limits
        self.max_depth = max_depth
        self.df = self._initialize_dataframe(max_depth, state.traderData)

    def _initialize_dataframe(self, max_depth, trader_data: Optional[str]) -> pd.DataFrame:
        """Load from traderData JSON or initialize an empty DataFrame."""
        if trader_data:
            try:
                data = json.loads(trader_data)
                return pd.DataFrame.from_dict(data) if data else self._empty_dataframe(max_depth)
            except json.JSONDecodeError:
                return self._empty_dataframe(max_depth)
        return self._empty_dataframe(max_depth)

    def _empty_dataframe(self, max_depth: int) -> pd.DataFrame:
        """Return an empty DataFrame with dynamically generated bid/ask columns based on user-defined max depth."""

        # Generate bid and ask price/volume columns dynamically based on max_depth
        bid_price_columns = [f"bid_price_{i+1}" for i in range(max_depth)]
        bid_volume_columns = [f"bid_volume_{i+1}" for i in range(max_depth)]
        ask_price_columns = [f"ask_price_{i+1}" for i in range(max_depth)]
        ask_volume_columns = [f"ask_volume_{i+1}" for i in range(max_depth)]

        # Base columns + dynamically generated bid/ask columns
        columns = (["timestamp", "product"] + bid_price_columns + bid_volume_columns + ask_price_columns + ask_volume_columns + [ 
                    "mid_price", "current_position", "max_sell_position", "max_buy_position", 
                    "observation_value", "conversion_bid", "conversion_ask", "transport_fees", "export_tariff",
                    "import_tariff", "sugar_price", "sunlight_index"])

        return pd.DataFrame(columns=columns)

    def update_new_state(self):
        """
        Extract data from the TradingState and append it to the DataFrame.
        """
        new_rows = []

        for product, order_depth in self.state.order_depths.items():
            # Extract all bid and ask prices with volumes dynamically
            bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)  # Highest first
            bid_volumes = [order_depth.buy_orders[price] for price in bid_prices]
            ask_prices = sorted(order_depth.sell_orders.keys())  # Lowest first
            ask_volumes = [order_depth.sell_orders[price] for price in ask_prices]

            # Now, manipulate and extract the bid/ask prices into separate entries
            bid_ask_price_volumes = {} 
            for i in range(self.max_depth):
                # Populate bid and ask prices based on available data
                bid_ask_price_volumes[f"bid_price_{i+1}"] = bid_prices[i] if i < len(bid_prices) else None
                bid_ask_price_volumes[f"ask_price_{i+1}"] = ask_prices[i] if i < len(ask_prices) else None
                bid_ask_price_volumes[f"bid_volume_{i+1}"] = bid_volumes[i] if i < len(bid_volumes) else None
                bid_ask_price_volumes[f"ask_volume_{i+1}"] = ask_volumes[i] if i < len(ask_volumes) else None

            # for easyness get best bid and ask
            best_bid = bid_prices[0] if bid_prices else None
            best_ask = ask_prices[0] if ask_prices else None

            # Compute mid price if both bids and asks are available
            mid_price = (best_bid + best_ask) / 2 if bid_prices and ask_prices else None

            # position and position limits
            max_sell_position = -1*self.position_limits[product]
            max_buy_position = self.position_limits[product]
            current_position = self.state.position.get(product, 0)

            # Observations
            observation_value = self.state.observations.plainValueObservations.get(product, None)

            # Conversion observations
            conv_obs = self.state.observations.conversionObservations.get(product, None)
            conversion_bid = conv_obs.bidPrice if conv_obs else None
            conversion_ask = conv_obs.askPrice if conv_obs else None
            transport_fees = conv_obs.transportFees if conv_obs else None
            export_tariff = conv_obs.exportTariff if conv_obs else None
            import_tariff = conv_obs.importTariff if conv_obs else None
            sugar_price = conv_obs.sugarPrice if conv_obs else None
            sunlight_index = conv_obs.sunlightIndex if conv_obs else None

            new_row = {
                "timestamp": self.state.timestamp,
                "product": product,
                "mid_price": mid_price,
                "max_sell_position": max_sell_position,
                "max_buy_position": max_buy_position,
                "current_position": current_position,
                "observation_value": observation_value,
                "conversion_bid": conversion_bid,
                "conversion_ask": conversion_ask,
                "transport_fees": transport_fees,
                "export_tariff": export_tariff,
                "import_tariff": import_tariff,
                "sugar_price": sugar_price,
                "sunlight_index": sunlight_index
            }
            new_row.update(bid_ask_price_volumes)


            new_rows.append(new_row)

        # Append new data to the DataFrame
        self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)

    def add_indicator(self, product: str, indicator_name: str, value):
        """
        Add a new calculated indicator to the latest row for a specific product.
        If the indicator doesn't exist, create a new column dynamically.
        """
        if indicator_name not in self.df.columns:
            self.df[indicator_name] = None

        # Locate the most recent row for the given product and update the indicator value
        mask = self.df["product"] == product
        if not mask.any():
            return  # No row exists yet for this product

        self.df.loc[mask.idxmax(), indicator_name] = value  # Update the latest row

    def save_to_trader_data(self):
        """Convert the DataFrame to a JSON string for storing in state.traderData."""
        return json.dumps(self.df.to_dict())

    def to_dataframe(self):
        """Return the stored trading data as a DataFrame."""
        return self.df


if __name__ == "__main__":
    state = get_state()
    state.traderData = ""
    td = TradingData(state, {"KELP": 50, "RAINFOREST_RESIN": 50},3)
    # print(td.to_dataframe().head(5))
    state.traderData = td.save_to_trader_data()
    # print(state.traderData)
    ## new iteration
    td = TradingData(state, {"KELP": 50, "RAINFOREST_RESIN": 50},3)
    td.update_new_state()
    print(td.to_dataframe().head(5))

