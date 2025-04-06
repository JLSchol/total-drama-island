from datamodel import TradingState, Order
from typing import Dict, List, Optional
import json
import numpy as np

import time
# Helper function to serialize numpy arrays
def numpy_array_default(obj):
    """Custom default function for serializing numpy arrays (including 2D arrays)."""
    if isinstance(obj, np.ndarray):
        # For 2D arrays, ensure it's serialized as a list of lists
        return obj.tolist()  # Convert 1D arrays to lists
    raise TypeError(f"Type {type(obj)} not serializable")

def convert_lists_to_numpy(data: Dict[str, Dict]):
    """Convert lists in JSON back to numpy arrays (including 2D arrays)."""
    for product, product_data in data.items():
        for key, value in product_data.items():
            if isinstance(value, list):
                # Convert lists back to numpy arrays
                arr = np.array(value, dtype=np.float64)
                data[product][key] = arr
    return data

class TradingData:

    def __init__(self, state: TradingState, position_limits: Dict[str, int], max_order_count: int, max_history_len: int):
        self.position_limits = position_limits
        self.max_order_count = max_order_count
        self.max_history_len = max_history_len
        self.data = self._initialize_data(state, position_limits)

    def _truncate_array(self, array: np.ndarray) -> np.ndarray:
        """Helper method to pop the first element of the array and keep the last max_history_len elements."""
        if array.ndim == 1:
            if array.shape[0] > self.max_history_len:
                return array[1:]  # Remove the first element of a 1D array
        elif array.ndim == 2:
            if array.shape[0] > self.max_history_len:
                return array[1:, :]  # Remove the first row of a 2D array (keeping all columns)
        return array

    def _initialize_data(self, state: TradingState, position_limits: Dict[str, int]) -> Dict[str, Dict]:
        if state.traderData:
            try:
                data = json.loads(state.traderData)
                if data:
                    return self._update_new_state(convert_lists_to_numpy(data), state, position_limits)
            except json.JSONDecodeError:
                pass  
        return self._from_empty_data(state, position_limits)

    def _from_empty_data(self, state: TradingState, position_limits: Dict[str, int]) -> Dict[str, Dict]:
        return self._update_new_state({}, state, position_limits)

    def _ensure_product_data(self, data: Dict[str, Dict], product: str) -> None:
        if product not in data:
            data[product] = {
                "timestamp": np.array([]),
                "bid_prices": np.empty((0, self.max_order_count), dtype=np.float64),
                "bid_volumes": np.empty((0, self.max_order_count), dtype=np.float64),
                "ask_prices": np.empty((0, self.max_order_count), dtype=np.float64),
                "ask_volumes": np.empty((0, self.max_order_count), dtype=np.float64),
                "best_bid": np.array([]),
                "best_bid_volume": np.array([]),
                "best_ask": np.array([]),
                "best_ask_volume": np.array([]),
                "total_ask_volume": np.array([]),
                "total_bid_volume": np.array([]),
                "total_volume": np.array([]),
                "mid_price": np.array([]),
                "weighted_mid_price": np.array([]),
                "max_sell_position": np.array([]),
                "max_buy_position": np.array([]),
                "current_position": np.array([]),
                "observation_plain_value": np.array([]),
                "observation_bidPrice": np.array([]),
                "observation_askPrice": np.array([]),
                "observation_transportFees": np.array([]),
                "observation_exportTariff": np.array([]),
                "observation_importTariff": np.array([]),
                "observation_sugarPrice": np.array([]),
                "observation_sunlightIndex": np.array([]),
            }

    def _update_new_state(self, data: Dict[str, Dict[str, np.ndarray]], state: TradingState, position_limits: Dict[str, int]) -> Dict[str, Dict[str, np.ndarray]]:
        for product, order_depth in state.order_depths.items():
            # get sorted np arrays
            bid_prices, bid_volumes = self.sort_bid_orders(order_depth.buy_orders, self.max_order_count)
            ask_prices, ask_volumes = self.sort_ask_orders(order_depth.sell_orders, self.max_order_count)

            total_bid_volume = np.sum(bid_volumes, where=~np.isnan(bid_volumes))
            total_ask_volume = np.sum(ask_volumes, where=~np.isnan(ask_volumes))
            total_volume = total_bid_volume + np.abs(total_ask_volume)

            # best bid/asks
            best_bid, best_bid_volume = (bid_prices[0], bid_volumes[0]) if bid_prices.size > 0 else (np.nan, np.nan)
            best_ask, best_ask_volume = (ask_prices[0], ask_volumes[0]) if ask_prices.size > 0 else (np.nan, np.nan)

            # mid_price
            if best_bid is not None and best_ask is not None:
                mid_price = np.mean(np.array([best_bid, best_ask], dtype=np.float64))
            else:
                mid_price = np.nan

            # Calculate the volume weighted mid-price
            # Create a mask that identifies valid (non-nan) values for both prices and volumes
            valid_bid_mask = ~np.isnan(bid_prices)
            valid_ask_mask = ~np.isnan(ask_prices)
            # Concatenate only the valid values (non-nan) for prices and volumes
            all_prices = np.concatenate((bid_prices[valid_bid_mask], ask_prices[valid_ask_mask]))
            all_volumes = np.concatenate((bid_volumes[valid_bid_mask], np.abs(ask_volumes[valid_ask_mask])))  # Absolute volumes for asks
            # Calculate the weighted mid-price
            if all_prices.size > 0:
                weighted_mid_price = np.average(all_prices, weights=all_volumes)
            else:
                weighted_mid_price = np.nan  # Return NaN if there are no valid prices

            # positions
            position = int(state.position.get(product, 0))
            max_buy_position = position_limits[product]
            max_sell_position = -max_buy_position

            # ensure product data (if data is {})
            self._ensure_product_data(data, product)

            # Append new values to each field
            data[product]["timestamp"] = np.append(data[product]["timestamp"], state.timestamp)
            data[product]["bid_prices"] = np.vstack([data[product]["bid_prices"], bid_prices]) #2d array
            data[product]["bid_volumes"] = np.vstack([data[product]["bid_volumes"], bid_volumes]) #2d array
            data[product]["ask_prices"] = np.vstack([data[product]["ask_prices"], ask_prices]) #2d array
            data[product]["ask_volumes"] = np.vstack([data[product]["ask_volumes"], ask_volumes]) #2d array
            data[product]["best_bid"] = np.append(data[product]["best_bid"], best_bid)
            data[product]["best_bid_volume"] = np.append(data[product]["best_bid_volume"], best_bid_volume)
            data[product]["best_ask"] = np.append(data[product]["best_ask"], best_ask)
            data[product]["best_ask_volume"] = np.append(data[product]["best_ask_volume"], best_ask_volume)
            data[product]["total_ask_volume"] = np.append(data[product]["total_ask_volume"], total_ask_volume)
            data[product]["total_bid_volume"] = np.append(data[product]["total_bid_volume"], total_bid_volume)
            data[product]["total_volume"] = np.append(data[product]["total_volume"], total_volume)
            data[product]["mid_price"] = np.append(data[product]["mid_price"], mid_price)
            data[product]["weighted_mid_price"] = np.append(data[product]["weighted_mid_price"], weighted_mid_price)
            data[product]["max_sell_position"] = np.append(data[product]["max_sell_position"], max_sell_position)
            data[product]["max_buy_position"] = np.append(data[product]["max_buy_position"], max_buy_position)
            data[product]["current_position"] = np.append(data[product]["current_position"], position)

            if state.observations.plainValueObservations:
                data[product]["observation_plain_value"] = np.append(data[product]["observation_plain_value"], state.observations.plainValueObservations.get(product))

            if state.observations.conversionObservations:
                obs = state.observations.conversionObservations.get(product)
                if obs:
                    data[product]["observation_bidPrice"] = np.append(data[product]["observation_bidPrice"], obs.bidPrice)
                    data[product]["observation_askPrice"] = np.append(data[product]["observation_askPrice"], obs.askPrice)
                    data[product]["observation_transportFees"] = np.append(data[product]["observation_transportFees"], obs.transportFees)
                    data[product]["observation_exportTariff"] = np.append(data[product]["observation_exportTariff"], obs.exportTariff)
                    data[product]["observation_importTariff"] = np.append(data[product]["observation_importTariff"], obs.importTariff)
                    data[product]["observation_sugarPrice"] = np.append(data[product]["observation_sugarPrice"], obs.sugarPrice)
                    data[product]["observation_sunlightIndex"] = np.append(data[product]["observation_sunlightIndex"], obs.sunlightIndex)

            # Apply truncation for all arrays
            for key, value in data[product].items():
                if isinstance(value, np.ndarray):  # Check if the value is a numpy array
                    data[product][key] = self._truncate_array(value)

        return data
    
    def sort_bid_orders(self, buy_orders, max_order_count: int):
        """
        Sorts buy orders in descending price order and pads with np.nan if necessary.
        
        Args:
            buy_orders (dict): Dictionary of {price: volume}.
            max_order_count (int): Maximum expected number of buy orders.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sorted and padded price and volume arrays.
        """
        if not buy_orders:
            # Return padded arrays if no orders exist
            return np.full(max_order_count, np.nan, dtype=np.float64), np.full(max_order_count, np.nan, dtype=np.float64)
        
        # Convert the dictionary to numpy arrays
        prices = np.array(list(buy_orders.keys()), dtype=np.float64)  # Prices as a numpy array
        volumes = np.array(list(buy_orders.values()), dtype=np.float64)  # Volumes as a numpy array

        # Sort prices in descending order and reorder volumes accordingly
        sorted_indices = np.argsort(prices)[::-1]  # Sorted indices in descending order
        sorted_prices = prices[sorted_indices]
        sorted_volumes = volumes[sorted_indices]

        # Pad the arrays with np.nan if fewer than max_order_count
        prices_padded = np.full(max_order_count, np.nan, dtype=np.float64)
        volumes_padded = np.full(max_order_count, np.nan, dtype=np.float64)

        # Efficiently copy the sorted values into the pre-allocated arrays
        prices_padded[:len(sorted_prices)] = sorted_prices
        volumes_padded[:len(sorted_volumes)] = sorted_volumes

        return prices_padded, volumes_padded
    
    def sort_ask_orders(self, sell_orders, max_order_count: int):
        """
        Sorts sell orders in ascending price order and pads with np.nan if necessary.
        
        Args:
            sell_orders (dict): Dictionary of {price: volume}.
            max_order_count (int): Maximum expected number of sell orders.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Sorted and padded price and volume arrays.
        """
        if not sell_orders:
            # Return padded arrays if no orders exist
            return np.full(max_order_count, np.nan, dtype=np.float64), np.full(max_order_count, np.nan, dtype=np.float64)
        
        # Convert the dictionary to numpy arrays for faster sorting
        prices = np.array(list(sell_orders.keys()), dtype=np.float64)  # Prices as a numpy array
        volumes = np.array(list(sell_orders.values()), dtype=np.float64)  # Volumes as a numpy array

        # Sort prices in ascending order and reorder volumes accordingly
        sorted_indices = np.argsort(prices)  # Sorted indices in ascending order
        sorted_prices = prices[sorted_indices]
        sorted_volumes = volumes[sorted_indices]

        # Pad the arrays with np.nan if fewer than max_order_count
        prices_padded = np.full(max_order_count, np.nan, dtype=np.float64)
        volumes_padded = np.full(max_order_count, np.nan, dtype=np.float64)

        # Efficiently copy the sorted values into the pre-allocated arrays
        prices_padded[:len(sorted_prices)] = sorted_prices
        volumes_padded[:len(sorted_volumes)] = sorted_volumes

        return prices_padded, volumes_padded

    def get_latest_fields(self, product: str) -> Dict[str, np.ndarray]:
        if product not in self.data:
            return {}

        latest_entry = {}
        for key, values in self.data[product].items():
            if isinstance(values, np.ndarray):
                if values.size == 0:
                    latest_entry[key] = np.nan  # Return NaN if empty
                elif values.ndim == 2 and values.shape[0]>0: # and values.shape[0]>0:
                    latest_entry[key] = values[-1, :]  # Get the last row for 2D arrays
                else:
                    latest_entry[key] = values[-1]  # Get the last element for 1D arrays
            else:
                raise ValueError(f"was expecting a np.array but got {values} of type {type(values)}")
        return latest_entry

    def get_field(self, product:str, field: str) -> Optional[np.ndarray]:
        if product not in self.data:
            return None  # Or handle this as needed

        if field not in self.data[product]:
            return None  # Handle missing field gracefully

        return self.data[product][field]
    
    def get_value_by_index(self, product: str, field: str, index: int) -> any:

        values = self.get_field(product, field)

        if values is None:
            return None  

        if isinstance(values, np.ndarray):
            if values.ndim == 1:
                return values[index]
            elif values.ndim == 2 and values.shape[0]>0: # and values.shape[0]>0:
                return values[index, :]
            else:
                raise ValueError(f"was expecting a np.array of ndim 1 or 2") 
        else:
            raise ValueError(f"was expecting a np.array but got {values} of type {type(values)}")
  
    def get_last_field(self, product: str, field: str) -> Optional[np.ndarray]:
        return self.get_value_by_index(product, field, -1)  
  
    def apply_indicator(self, product: str, indicator_name: str, value: np.float64):    
        if indicator_name in ["bid_prices","bid_volumes","ask_prices","ask_volumes"]:
            self.apply_2d_indicator(product, indicator_name, value)

        if product not in self.data:
            self.data[product] = {}
        
        if indicator_name not in self.data[product]:
            self.data[product][indicator_name] = np.array([])

        if value is not None:
            self.data[product][indicator_name] = np.append(self.data[product][indicator_name], value)
            self.data[product][indicator_name] = self._truncate_array(self.data[product][indicator_name])

    def apply_2d_indicator(self, product: str, indicator_name: str, values: np.ndarray):
        if product not in self.data:
            self.data[product] = {}
        
        if indicator_name not in self.data[product]:
            self.data[product][indicator_name] = np.array([])

        if values:
            # Check if the 2D array already has data, then append the new row
            if self.data[product][indicator_name].size == 0:
                # If the array is empty, initialize it with the new values as a single row
                self.data[product][indicator_name] = np.array([values])
            else:
                # If the array has data, append the new values as a new row
                self.data[product][indicator_name] = np.vstack([self.data[product][indicator_name], values])
                self.data[product][indicator_name] = self._truncate_array(self.data[product][indicator_name])

    def get_data_as_json(self) -> str:
        # Serialize the data to JSON
        return json.dumps(self.data, default=numpy_array_default)

    def print_sim_step_data(self):
        data = {}
        for product in self.data.keys():
            product_data = self.get_latest_fields(product)
            data[product] = product_data
        # convert to json string, such that it can be loaded easily later
        print(json.dumps(data, default=numpy_array_default))

#TODO: CONVERT TO NUMPY ARRAY STUFFIES
def is_available(best: Optional[float], best_amount: Optional[int]) -> bool:
    return best is not None and best_amount is not None

def adjust_sell_quantity(proposed_sell_quantity: int, max_sell_limit: int, current_position: int):
    if max_sell_limit >= 0 or proposed_sell_quantity >= 0:
        raise ValueError(f"{proposed_sell_quantity=} or {max_sell_limit=}, should be negative for selling")
    
    max_allowed_sell_quantity = max_sell_limit - current_position
    if proposed_sell_quantity < max_allowed_sell_quantity:
        adjusted_sell_quantity = max(proposed_sell_quantity, max_allowed_sell_quantity)
        remaining_sell_capacity = 0
    else:
        adjusted_sell_quantity = proposed_sell_quantity
        remaining_sell_capacity = max_allowed_sell_quantity - proposed_sell_quantity
    return adjusted_sell_quantity, remaining_sell_capacity

def adjust_buy_quantity(proposed_buy_quantity: int, max_buy_limit: int, current_position: int):
    if max_buy_limit <= 0 or proposed_buy_quantity <= 0:
        raise ValueError(f"{proposed_buy_quantity=} or {max_buy_limit=}, should be positive for buying")
    
    max_allowed_buy_quantity = max_buy_limit - current_position
    if proposed_buy_quantity > max_allowed_buy_quantity:
        adjusted_buy_quantity = min(proposed_buy_quantity, max_allowed_buy_quantity)
        remaining_buy_capacity = 0
    else:
        adjusted_buy_quantity = proposed_buy_quantity
        remaining_buy_capacity = max_allowed_buy_quantity - proposed_buy_quantity
    return adjusted_buy_quantity, remaining_buy_capacity

def get_best_order(order_type: str, product: str, price: float, amount: int, current_position: int, max_position: int, orders: List[Order]) -> List[Order]:
    """
    Creates a buy or sell order based on the best available price and quantity.
    
    :param order_type: "buy" for buying, "sell" for selling
    :param product: The product being traded
    :param price: The best price available
    :param amount: The quantity available (negative for sell, positive for buy)
    :param current_position: The trader's current position
    :param max_position: The trader's maximum allowable position
    :param orders: The list of existing orders to append to
    :return: The updated list of orders
    """
    if not is_available(price, amount):
        return orders, None  # No valid price or quantity available

    # Flip sign: ask/sell is negative, bid/buy is positive
    order_quantity = -amount

    # Adjust based on position limits
    if order_type == "buy":
        order_quantity, remaining_capacity = adjust_buy_quantity(order_quantity, max_position, current_position)
        if order_quantity <= 0:
            return orders, remaining_capacity  # No valid buy order
    else:  # Sell order
        order_quantity, remaining_capacity = adjust_sell_quantity(order_quantity, max_position, current_position)
        if order_quantity >= 0:
            return orders, remaining_capacity  # No valid sell order

    # Append the order
    orders.append(Order(product, price, order_quantity))
    return orders, remaining_capacity

def get_best_orders(product: str, price: float, 
                    best_ask: int, best_ask_amount: int, 
                    best_bid: int, best_bid_amount: int, 
                    current_position: int, max_position: int, orders: List[Order]) -> List[Order]:

    if best_ask is not None and best_ask < price:
        orders, _ = get_best_order("buy", product, best_ask, best_ask_amount, current_position, max_position, orders)
    # Place sell order if best bid is higher than the fair price
    if best_bid is not None and best_bid > price:
        orders, _ = get_best_order("sell", product, best_bid, best_bid_amount, current_position, -max_position, orders)
    return orders

def sma(prices: np.ndarray, window: int) -> float:
    # Check for valid input
    if prices.size == 0 or window <= 0:
        raise ValueError("Prices array is empty or window size is invalid")
    
    if prices.size == 1:
        return prices[0]

    # Adjust window size if it's larger than available prices
    window = min(window, prices.size)

    # Get the last `window` prices, ignoring NaNs
    valid_prices = prices[-window:]
    
    # Calculate the simple moving average, ignoring NaNs
    sma = np.nanmean(valid_prices)  # This will ignore NaN values when calculating the mean
    
    return sma

def sigmoid(x: float, alpha: float):
    # alpha is a steepness parameter 2 is steep and 0.5
    return 1 / (1 + np.exp(-alpha * x))

def calculate_trend_score(current_price: int, sma_small:float, alpha:float):
    """
    Calculate a trend score mapped to [-1, 1] based on price deviation from sma_small.
    """
    price_diff_small = current_price - sma_small
    
    # Apply the sigmoid function and scale to map to [-1, 1]
    trend_score = 2*(sigmoid(price_diff_small, alpha) - 0.5)

    return trend_score

def sma_crossover_score(td: TradingData, product: str, 
                        prices:list[float], 
                        window_small: int, window_large: int, sigmoid_alpha: float):
    # returns a trend score between [-1, 1]
    # if there is a crossover, it returns [0, 1] for bullish and [0, -1] for bearish
    # depending on the price action associated with the crossover
    if len(prices) == 0 or window_small <= 0 or window_small <= 0:
        raise ValueError("Prices list is empty or window size is invalid value <=0")

    # init score (no trend):
    trend_score = 0

    # define names
    sma_small_name = f"sma_cross_{window_small}"
    sma_large_name = f"sma_cross_{window_large}"

    # get previous sma step
    previous_sma_small = td.get_last_field(product, sma_small_name)
    previous_sma_large = td.get_last_field(product, sma_large_name)

    # if they did not exist yet, create them and initialize them with the latest price
    if previous_sma_small is None and previous_sma_large is None:
        td.apply_indicator(product, sma_small_name, prices[-1])
        td.apply_indicator(product, sma_large_name, prices[-1])
        td.apply_indicator(product, "trend_score", trend_score)
        return trend_score

    # calculate current step    
    current_price = prices[-1]
    current_sma_small = sma(prices, window_small)
    current_sma_large = sma(prices, window_large)

    # update the sma indicators
    td.apply_indicator(product, sma_small_name, current_sma_small)
    td.apply_indicator(product, sma_large_name, current_sma_large)
    if len(prices) > window_large*.8:
        # from .8*the large window we start to make predictions
        td.apply_indicator(product, "trend_score", trend_score)
        return trend_score
    
    # calculate the trend score if a crossover happend
    if current_sma_small > current_sma_large and previous_sma_small <= previous_sma_large:
        # bullish cross return value between [0, 1] 
        trend_score = max( calculate_trend_score(current_price, current_sma_small, sigmoid_alpha), 0)

    elif current_sma_small < current_sma_large and previous_sma_small >= previous_sma_large:
        # bearish cross return value between [0, 1] 
        trend_score = min( calculate_trend_score(current_price, current_sma_small, sigmoid_alpha), 0)

    td.apply_indicator(product, "trend_score", trend_score)

    return trend_score

def order_imbalance_score(td: TradingData, product: str, 
                          total_ask_volume: int, total_bid_volume: int):
    """
    Computes an order imbalance score between -1 (selling pressure) and +1 (buying pressure).
    """
    imbalance_score = 0
    if total_ask_volume is None or total_bid_volume is None:
        td.apply_indicator(product, "imbalance_score", imbalance_score)        
        return imbalance_score
    
    if total_ask_volume == 0 and total_ask_volume == 0:
        td.apply_indicator(product, "imbalance_score", imbalance_score)        
        return imbalance_score

    nominator = np.abs(total_bid_volume) - np.abs(total_ask_volume)
    denominator = np.abs(total_bid_volume) + np.abs(total_ask_volume)

    if denominator == 0:
        td.apply_indicator(product, "imbalance_score", imbalance_score)    
        return imbalance_score
    
    imbalance_score = nominator / denominator
    td.apply_indicator(product, "imbalance_score", imbalance_score)

    return imbalance_score

def spread_to_price_ratio(td: TradingData, product: str, 
                          best_ask: int, best_bid: int, mid_price: float):
    """
    Computes the spread-to-price ratio, indicating market liquidity.
    """
    spread_ratio = 0
    if None in (best_ask, best_bid, mid_price):
        td.apply_indicator(product, "spread_ratio", spread_ratio)    
        return spread_ratio
    
    if best_bid == 0 and best_ask == 0:
        td.apply_indicator(product, "spread_ratio", spread_ratio)    
        return spread_ratio

    # calculate the spread
    spread = best_ask - best_bid
    spread_ratio = spread / mid_price if mid_price != 0 else 0
    td.apply_indicator(product, "spread_ratio", spread_ratio)    

    return spread_ratio

def calculate_dynamic_spread(td: TradingData, product: str, 
                             spread_ratio: float, spread_scaling: float,
                             trend_score: float, trend_scaling: float,
                             pressure_score: float, pressure_scaling: float,
                             base_spread: int, min_spread_factor: float =0.5, max_spread_factor: float =2):
    # TODO: create dynamic scaling params (spread, trend, pressure, min/max spread)
    """
    Computes the dynamic spread based on market conditions.

    Parameters:
    - spread_ratio (float): Spread-to-price ratio (higher = less liquidity).
    - trend_score (float): Trend signal between [-1, 1], where 1 is strongly bullish.
    - pressure_score (float): Order imbalance score between [-1, 1], where 1 is strong buying pressure.
    - base_spread (float): The normal market spread. pick a value halfway between the observed spread range
        - KELP: observed (bid ask) spread [1,4] -> suggested base spread 2
        - RAINFOREST RESIN: observed (bid ask) spread [2,10] -> suggested base spread 5
    Returns:
    - dynamic_spread (float): Adjusted spread based on market conditions. clamped between half and twice the base spread
    """
    if base_spread is None or base_spread < 0:
        raise ValueError("base_spread should be >= 0")

    # Adjust spread based on liquidity (spread ratio)
    # spread_scaling = 1
    liquidity_adjustment = spread_ratio * spread_scaling * base_spread  # probl small increase or close to 0

    # Adjust spread based on market trend (bullish reduces ask spread, bearish reduces bid spread)
    # trend_scaling = .2 #trend_scaling is between [-1,1]: 
    trend_adjustment = -trend_score * trend_scaling * base_spread  # added range [-.2; .2]

    # Adjust spread based on market pressure (more bids → tighten bid, more asks → tighten ask)
    # pressure_scaling = .3 #pressure_score is between [-1,1]
    pressure_adjustment = -pressure_score * pressure_scaling * base_spread  # added range [-.3; .3] (sligly more important then trend_adjustment)

    # Compute final dynamic spread
    dynamic_spread = base_spread + liquidity_adjustment + trend_adjustment + pressure_adjustment  

    # Ensure spread stays within min and max limits
    def clamp(value, min_value, max_value):
        """Ensures value stays within defined bounds."""
        return max(min_value, min(value, max_value))
    
    # Define minimum and maximum spread limits to avoid extreme values
    min_spread = base_spread * min_spread_factor   # Spread shouldn't be too small
    max_spread = base_spread * max_spread_factor   # Spread shouldn't be too large
    dynamic_spread = clamp(dynamic_spread, min_spread, max_spread)
    td.apply_indicator(product, "dynamic_spread", dynamic_spread)

    return dynamic_spread

def calculate_order_prices(td: TradingData, product: str, fair_price: float, dynamic_spread: float, pressure_score: float, shift_alpha: float = 0.2):
    """
    Calculates buy and sell order prices based on fair price, spread, and market pressure.
    """
    shift = pressure_score * shift_alpha * dynamic_spread
    buy_price = round(fair_price - (dynamic_spread / 2) + shift) 
    sell_price = round(fair_price + (dynamic_spread / 2) + shift)

    td.apply_indicator(product, "calculated_shift", shift)
    td.apply_indicator(product, "calculated_buy", buy_price)
    td.apply_indicator(product, "calculated_sell", sell_price)
    
    return buy_price, sell_price

def get_orders(td: TradingData, product: str, orders: list[Order], 
               fair_price: float, best_bid: int, best_ask: int, 
               dynamic_spread: float, pressure_score: float, 
               best_bid_amount: int, best_ask_amount: int, 
               current_position: int, max_position: int,
               position_threshold_factor: float, shift_alpha: float, order_volume_factor: float):
    """
    Places buy and sell orders based on fair price, dynamic spread, trend, and market pressure.
    """
    # 1. Ensure required data is available
    # otherwise just buy best ask en sell best bid
    if None in (fair_price, best_bid, best_ask, dynamic_spread, pressure_score):
        orders, _ = get_best_order("sell", product, best_bid, best_bid_amount, current_position, -max_position, orders)
        orders, _ = get_best_order("buy", product, best_ask, best_ask_amount, current_position, max_position, orders)
        return orders  

    # 2. Rebalance if position is near limits
    position_threshold = position_threshold_factor * max_position
    if abs(current_position) >= abs(position_threshold):
        if current_position > 0:
            orders, _ = get_best_order("sell", product, best_bid, best_bid_amount, current_position, -max_position, orders)
        else:
            orders, _ = get_best_order("buy", product, best_ask, best_ask_amount, current_position, max_position, orders)
        return orders

    
    # 3. calculate Buy & Sell Prices (ensure integers)
    buy_price, sell_price = calculate_order_prices(td, product, fair_price, dynamic_spread, pressure_score, shift_alpha)

    # 4. Ensure Orders Stay Competitive
    buy_price = round(min(buy_price, best_bid))   # Don't bid above best bid
    sell_price = round(max(sell_price, best_ask)) # Don't sell below best ask

    # 5. Set Order Volumes
    base_volume = order_volume_factor*max_position # ideally dynamically calculate (order spikes, order book depth, volatility)
    buy_volume = round(base_volume * (1 + pressure_score))  # Increase if buy pressure is high
    sell_volume = round(base_volume * (1 - pressure_score)) # Reduce if sell pressure is high
    td.apply_indicator(product, "calculated_base_volume", base_volume)
    td.apply_indicator(product, "calculated_buy_volume", buy_volume)
    td.apply_indicator(product, "calculated_sell_volume", sell_volume)

    # 6. Ensure minimum order size of 1
    buy_volume = int(max(1, buy_volume))
    sell_volume = int(max(1, sell_volume))

    td.apply_indicator(product, "market_order_buy_price", buy_price)
    td.apply_indicator(product, "market_order_buy_volume", buy_volume)
    td.apply_indicator(product, "market_order_sell_price", sell_price)
    td.apply_indicator(product, "market_order_sell_volume", -sell_volume)
    td.apply_indicator(product, "market_order_buy_price", buy_price)

    # 5. Place Orders
    orders.append(Order(product, buy_price, buy_volume))
    orders.append(Order(product, sell_price, -sell_volume))

    return orders

def market_maker_strategy(td: TradingData, product: str, orders: list[Order], 
                          fair_price_window: int, shift_alpha: float,
                          sma_small_window: int, sma_large_window: int, sigmoid_alpha: float, 
                          spread_scaling: float, trend_scaling: float, pressure_scaling: float,
                          base_spread: int, min_spread_factor: float, max_spread_factor: float,
                          max_position: int, position_threshold_factor: float, order_volume_factor: float):
    """
    Market-making strategy that places orders based on fair price, trend, and order imbalance.
    """
    # 1. compute fair price
    mid_prices = td.get_field(product, "mid_price")
    fair_price = sma(mid_prices, fair_price_window)
    td.apply_indicator(product, "fair_price", fair_price)

    # 2. Identify market trend returns [-1 and 1]
    trend_score = sma_crossover_score(td, product, mid_prices, sma_small_window, sma_large_window, sigmoid_alpha)

    # 3. Analyze market pressure
    total_ask_volume = td.get_last_field(product, "total_ask_volume")
    total_bid_volume = td.get_last_field(product, "total_bid_volume")
    pressure_score = order_imbalance_score(td, product, total_ask_volume, total_bid_volume)

    # 4 liquidity
    best_bid = td.get_last_field(product, "best_bid")
    best_ask = td.get_last_field(product, "best_ask")
    spread_ratio = spread_to_price_ratio(td, product, best_ask, best_bid, mid_prices[-1])

    # 5. calculate dynamic Spread based on market conditions
    dynamic_spread = calculate_dynamic_spread(td, product, 
                             spread_ratio, spread_scaling,
                             trend_score, trend_scaling,
                             pressure_score, pressure_scaling,
                             base_spread, min_spread_factor, max_spread_factor)

    # 6 get orders
    best_bid_amount = td.get_last_field(product, "best_bid_amount")
    best_ask_amount = td.get_last_field(product, "best_ask_amount")
    current_position = td.get_last_field(product, "current_position")
    sim_step = td.get_last_field(product, "timestamp")/100

    # if we do not have calculated enough data for our strategy, just buy the best ask and sell best bid
    if sim_step < sma_large_window:
        return get_best_orders(product, fair_price, 
                        best_ask, best_ask_amount, best_bid, best_bid_amount,
                        current_position, max_position, orders)

    orders = get_orders(td, product, orders, 
               fair_price, best_bid, best_ask, 
               dynamic_spread, pressure_score, 
               best_bid_amount, best_ask_amount, 
               current_position, max_position,
               position_threshold_factor, shift_alpha, order_volume_factor)

    return orders


class Trader:
    def run(self, state: TradingState):
        # start_time = time.time()  # Start the timer
        result = {}
        conversions = 0
        position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        max_order_count = 4 #  order book depth + 1 = 4
        max_history = 55

        td = TradingData(state, position_limits, max_order_count, max_history)

        for product in state.order_depths:
            orders: List[Order] = []

            if product == "KELP":
                orders = market_maker_strategy(td, product, orders,
                                    fair_price_window=5, shift_alpha=0.25,
                                    sma_small_window=8, sma_large_window=50, sigmoid_alpha=0.4,
                                    spread_scaling=1, trend_scaling=0.3, pressure_scaling=0.4,
                                    base_spread=2, min_spread_factor=0.5, max_spread_factor=2,
                                    max_position=position_limits[product], 
                                    position_threshold_factor=0.75, order_volume_factor=0.15)

            if product == "RAINFOREST_RESIN":
                orders = market_maker_strategy(td, product, orders,
                                fair_price_window=5, shift_alpha=.25,
                                sma_small_window= 8, sma_large_window=50, sigmoid_alpha=.4,
                                spread_scaling=1, trend_scaling=0.2, pressure_scaling=0.3,
                                base_spread=5, min_spread_factor=0.5, max_spread_factor=2,
                                    max_position=position_limits[product], 
                                    position_threshold_factor=0.8, order_volume_factor=0.1)
            
            result[product] = orders

        td.print_sim_step_data()

        traderData = td.get_data_as_json()

        # end_time = time.time()  # End the timer
        # execution_time = end_time - start_time  # Calculate the elapsed time
        # print(f"Execution Time: {execution_time:.4f} seconds")

        return result, conversions, traderData

# from mock import state, state2
# if __name__ == "__main__":
#     trader = Trader()
#     newstate = state
#     for i in range(2000):
        
#         result,conversions, data = trader.run(newstate)
#         newstate.traderData = data



    # max_order_count = 5  # Define maximum expected order depth
    # position_limits = {"RAINFOREST_RESIN":50,"KELP":50}

    # td = TradingData(state, position_limits, max_order_count)

    # trader = Trader()
    # result,conversions, data = trader.run(state)
    # # print(data)
    # print()
    # print()
    # print()
    # print("--- NEW STEP ---") 
    # print()
    # print()
    # print()
    # state2.traderData = data
    # result,conversions, data = trader.run(state2)
    # print(data)
