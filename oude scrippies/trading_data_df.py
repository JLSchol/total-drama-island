import pandas as pd
import json
from typing import Dict, List, Optional
from datamodel import TradingState


class TradingData:
    def __init__(self, state: TradingState, position_limits: dict[str,int], max_depth: int=3):
        """
        Initialize TradingData from the state object.

        This constructor initializes the TradingData class by setting up the initial state,
        position limits, and maximum depth for order book analysis. It also initializes
        the DataFrame that will store the trading data.

        Parameters:
        state : object
            The current trading state object containing market information.
        position_limits : dict[str,int]
            A dictionary mapping product names to their respective position limits.
        max_depth : int, optional
            The maximum depth of the order book to consider (default is 3).

        Returns:
        None

        Note:
        The method initializes the DataFrame by calling the _initialize_dataframe method.
        If traderData exists in the state, it loads it into a DataFrame.
        Otherwise, it starts from an empty DataFrame.
        """
        # self.state = state
        self.position_limits = position_limits
        self.max_depth = max_depth

        self.df = self._initialize_dataframe(max_depth, state, position_limits)

    def _initialize_dataframe(self, max_depth: int, state: TradingState, position_limits: dict[str,int]) -> pd.DataFrame:
        """
        Initialize the DataFrame either from existing traderData or create a new empty DataFrame.

        This method attempts to load data from the state's traderData JSON. If successful, it
        returns a DataFrame created from this data. If the traderData is empty or there's a
        JSON decoding error, it initializes and returns an empty DataFrame.

        Parameters:
        max_depth (int): The maximum depth of order book to consider.
        state (object): The current trading state object containing market information.
        position_limits (dict[str,int]): A dictionary mapping product names to their respective position limits.

        Returns:
        pd.DataFrame: A DataFrame containing the trading data, either loaded from existing data
                      or newly initialized if no valid data exists.

        Raises:
        JSONDecodeError: If there's an error in decoding the JSON from traderData.
        """
        if state.traderData:
            try:
                data = json.loads(state.traderData)
                if data:
                    df = pd.DataFrame.from_dict(data)
                    return self._update_new_state(df, state, position_limits)
                else:
                    return self._from_empty_dataframe(max_depth, state, position_limits)
            except json.JSONDecodeError as e:
                print(f"reinit dataframe because of JSON decoding error: {e}")
                return self._from_empty_dataframe(max_depth, state, position_limits)
        return self._from_empty_dataframe(max_depth, state, position_limits)

    def _from_empty_dataframe(self, max_depth: int, state: TradingState, position_limits: dict[str,int]) -> pd.DataFrame:
        """
        Create and return an empty DataFrame with dynamically generated bid/ask columns.

        This function initializes an empty DataFrame with columns for bid and ask prices
        and volumes, based on the specified maximum depth. It also includes additional
        columns for various trading metrics and observations. The DataFrame is then
        populated with initial data from the provided state. This is considered the 
        absolute basics that are required. Other indicators can be added later.

        Parameters:
        max_depth (int): The maximum depth of the order book to consider for generating
                         bid and ask columns.
        state: The current trading state object containing market information.
        position_limits (dict[str,int]): A dictionary mapping product names to their
                                         respective position limits.

        Returns:
        pd.DataFrame: An initialized DataFrame with columns for trading data, populated
                      with initial data from the state. 
        """
        print("create empty dataframe")

        # Generate bid and ask price/volume columns dynamically based on max_depth
        bid_price_columns = [f"bid_price_{i+1}" for i in range(max_depth)]
        bid_volume_columns = [f"bid_volume_{i+1}" for i in range(max_depth)]
        ask_price_columns = [f"ask_price_{i+1}" for i in range(max_depth)]
        ask_volume_columns = [f"ask_volume_{i+1}" for i in range(max_depth)]

        # Base columns + dynamically generated bid/ask columns
        columns = (["timestamp", "product"] + bid_price_columns + bid_volume_columns + ask_price_columns + ask_volume_columns + [ 
                    "best_bid", "best_bid_volume", "best_ask", "best_ask_volume", "mid_price", "current_position", "max_sell_position", "max_buy_position", 
                    "observation_value", "conversion_bid", "conversion_ask", "transport_fees", "export_tariff",
                    "import_tariff", "sugar_price", "sunlight_index"])

        df = pd.DataFrame(columns=columns) # create the headers
        df = self._update_new_state(df, state, position_limits) # populate with data from state
        return df

    def _update_new_state(self, df: pd.DataFrame, state: TradingState, position_limits: dict[str,int]):
        """
        Update the DataFrame with new trading data extracted from the current state.

        This function processes the current trading state to extract relevant data
        for each product, including bid/ask prices and volumes, position limits,
        and various observations. It appends this data as new rows to the provided
        DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to be updated with new trading data.
        state: The current trading state object containing market information.
        position_limits (dict[str,int]): A dictionary mapping product names to their
                                         respective position limits.

        Returns:
        pd.DataFrame: The updated DataFrame with new rows appended, containing the
                      latest trading data extracted from the state.
        """
        new_rows = []

        for product, order_depth in state.order_depths.items():
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
            best_bid_volume = bid_volumes[0] if bid_volumes else None
            best_ask_volume = ask_volumes[0] if ask_volumes else None

            # Compute mid price if both bids and asks are available
            mid_price = (best_bid + best_ask) / 2 if bid_prices and ask_prices else None

            # position and position limits
            max_sell_position = -1*position_limits[product]
            max_buy_position = position_limits[product]
            current_position = state.position.get(product, 0)

            # Observations
            observation_value = state.observations.plainValueObservations.get(product, None)

            # Conversion observations
            conv_obs = state.observations.conversionObservations.get(product, None)
            conversion_bid = conv_obs.bidPrice if conv_obs else None
            conversion_ask = conv_obs.askPrice if conv_obs else None
            transport_fees = conv_obs.transportFees if conv_obs else None
            export_tariff = conv_obs.exportTariff if conv_obs else None
            import_tariff = conv_obs.importTariff if conv_obs else None
            sugar_price = conv_obs.sugarPrice if conv_obs else None
            sunlight_index = conv_obs.sunlightIndex if conv_obs else None

            new_row = {
                "timestamp": state.timestamp,
                "product": product,
                "best_bid": best_bid,
                "best_bid_volume": best_bid_volume,
                "best_ask": best_ask,
                "best_ask_volume": best_ask_volume,
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
        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    def apply_indicator(self, product: str, indicator_name: str, value):
        """
        Add a new calculated indicator + value to the latest row for a specific product.
        If the indicator doesn't exist, create a new column dynamically.

        Parameters:
        product (str): The name of the product for which the indicator is to be added.
        indicator_name (str): The name of the indicator to be added.
        value: The value of the indicator to be added to the DataFrame.

        Returns:
        None: This function does not return a value. It updates the DataFrame in place.
        """
        # if indicator has not a colum yet, create a new column
        if indicator_name not in self.df.columns:
            self.df[indicator_name] = None

        # Locate the most recent row for the given product and update the indicator value
        mask = self.df["product"] == product # boolean mask returning true for matching product
        if not mask.any():
            return  # No row exists yet for this product

        # Update the last occurrence (most recent row added appended to df)
        self.df.loc[mask[::-1].idxmax(), indicator_name] = value

    def get_df_as_json_string(self):
        """Convert the DataFrame to a JSON string for storing in state.traderData."""
        return json.dumps(self.df.to_dict())

    def get_df(self):
        """Return the stored trading data as a DataFrame."""
        return self.df


if __name__ == "__main__":
    from mock import state, state2
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Auto-adjust width to fit the screen

    state.traderData = ""
    print(f"state: {state}")
    td = TradingData(state, {"KELP": 50, "RAINFOREST_RESIN": 50},3) # create new df
    print(td.get_df().head(5))
    print("----")
    new_trader_data = td.get_df_as_json_string()
    print(f"{new_trader_data=}")
    print()
    print("----")
    print("new iteration")
    print("----")
    print()
    # new iteration
    state2.traderData = new_trader_data
    td = TradingData(state2, {"KELP": 50, "RAINFOREST_RESIN": 50},3)
    print(td.get_df().head(5))

