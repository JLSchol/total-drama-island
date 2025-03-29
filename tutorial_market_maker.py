from datamodel import TradingState, Order
from typing import Dict, List
import json
import numpy as np

class TradingData:

    def __init__(self, state: TradingState, position_limits: Dict[str, int]):
        self.position_limits = position_limits
        self.data = self._initialize_data(state, position_limits)

    def _initialize_data(self, state: TradingState, position_limits: Dict[str, int]) -> Dict[str, Dict]:
        if state.traderData:
            try:
                data = json.loads(state.traderData)
                if data:
                    return self._update_new_state(data, state, position_limits)
            except json.JSONDecodeError:
                pass  
        return self._from_empty_data(state, position_limits)

    def _from_empty_data(self, state: TradingState, position_limits: Dict[str, int]) -> Dict[str, Dict]:
        return self._update_new_state({}, state, position_limits)

    def _update_new_state(self, data: Dict[str, Dict[str, List]], state: TradingState, position_limits: Dict[str, int]) -> Dict[str, Dict[str, List]]:
        for product, order_depth in state.order_depths.items():
            # Sort buy orders from highest to lowest price (best bid to worst bid)
            buy_orders = order_depth.buy_orders
            sell_orders = order_depth.sell_orders
            best_bid, _ = self.get_buy_order(buy_orders,0)
            best_ask, _ = self.get_sell_order(sell_orders,0)
            mid_price = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None

            # Ensure data structure is initialized
            if product not in data:
                data[product] = {
                    "timestamp": [],
                    "buy_orders": [],
                    "sell_orders": [],
                    "best_bid": [],
                    "best_bid_volume": [],
                    "best_ask": [],
                    "best_ask_volume": [],
                    "total_ask_volume": [],
                    "total_bid_volume": [],
                    "mid_price": [],
                    "max_sell_position": [],
                    "max_buy_position": [],
                    "current_position": [],
                    "observation_plain_value": [],
                    "observation_bidPrice": [],
                    "observation_askPrice": [],
                    "observation_transportFees": [],
                    "observation_exportTariff": [],
                    "observation_importTariff": [],
                    "observation_sugarPrice": [],
                    "observation_sunlightIndex": [],
                }

            # Append new values to each field
            data[product]["timestamp"].append(state.timestamp)
            data[product]["buy_orders"].append(dict(buy_orders))
            data[product]["sell_orders"].append(dict(sell_orders))
            data[product]["best_bid"].append(best_bid)
            data[product]["best_bid_volume"].append(buy_orders.get(best_bid, 0))
            data[product]["best_ask"].append(best_ask)
            data[product]["best_ask_volume"].append(sell_orders.get(best_ask, 0))
            data[product]["total_ask_volume"].append(sum(sell_orders.values()))
            data[product]["total_bid_volume"].append(sum(buy_orders.values()))
            data[product]["mid_price"].append(mid_price)
            data[product]["max_sell_position"].append(-position_limits[product])
            data[product]["max_buy_position"].append(position_limits[product])
            data[product]["current_position"].append(int(state.position.get(product, 0)))

            if state.observations.plainValueObservations:
                data[product]["observation_plain_value"].append(state.observations.plainValueObservations[product])

            if state.observations.conversionObservations:
                data[product]["observation_bidPrice"].append(state.observations.conversionObservations[product].bidPrice)
                data[product]["observation_askPrice"].append(state.observations.conversionObservations[product].askPrice)
                data[product]["observation_transportFees"].append(state.observations.conversionObservations[product].transportFees)
                data[product]["observation_exportTariff"].append(state.observations.conversionObservations[product].exportTariff)
                data[product]["observation_importTariff"].append(state.observations.conversionObservations[product].importTariff)
                data[product]["observation_sugarPrice"].append(state.observations.conversionObservations[product].sugarPrice)
                data[product]["observation_sunlightIndex"].append(state.observations.conversionObservations[product].sunlightIndex)

        return data

    def get_product_data(self, product: str) -> Dict:
        return self.data.get(product, {})

    def get_latest_fields(self, product: str) -> Dict[str, any]:
        if product not in self.data:
            return {}

        latest_entry = {}
        for key, values in self.data[product].items():
            if values:  # Ensure there's data
                # Get the most recent entry, This does not guarantee matching timestamp, 
                # with data, it just grabs the latest from whatever iteraction that is..
                latest_entry[key] = values[-1]  
        return latest_entry

    def get_last_field(self, product: str, field: str) -> any:
        return self.get_value_by_index(product, field, -1)
    
    def get_values_by_range(self, product: str, field: str, start: int, length: int) -> list:
        if product not in self.data:
            return []  # Return an empty list if the product doesn't exist

        if field not in self.data[product]:
            return []  # Return an empty list if the field doesn't exist

        values = self.data[product][field]

        # Adjust negative indices correctly
        if start < 0:
            start = len(values) + start  # Convert to a positive index if negative

        # Ensure we don't slice out of bounds
        if start < 0:
            start = 0  # If the start is still out of bounds, start from 0

        # If length extends beyond the end of the list, limit it to the list size
        end = min(start + length, len(values))

        # Return the sliced range
        return values[start:end]

    def get_field(self, product:str, field: str) -> any:
        if product not in self.data:
            return None  # Or handle this as needed

        if field not in self.data[product]:
            return None  # Handle missing field gracefully

        return self.data[product][field]

    def get_value_by_index(self, product: str, field: str, index: int) -> any:

        values = self.get_field(product, field)

        if values is None:
            return None  

        # Check if the index is valid (both positive and negative indices)
        if -len(values) <= index < len(values):
            return values[index]  # Return the value at the specified index (positive or negative)

        return None  # Return None if the index is out of bounds
  
    def get_value_by_timestamp(self, product: str, field: str, timestamp: int) -> any:
        if product not in self.data:
            return None  # Or handle this as needed

        if field not in self.data[product]:
            return None  # Handle missing field gracefully

        timestamps = self.data[product]["timestamp"]
        values = self.data[product][field]

        # Find the index of the matching timestamp
        if timestamp in timestamps:
            index = timestamps.index(timestamp)
            return values[index]  # Return the value at the matching timestamp index

        return None  # Return None if timestamp is not found

    def apply_indicator(self, product: str, indicator_name: str, value):
        if product not in self.data:
            self.data[product] = {}  # Ensure product exists

        if indicator_name not in self.data[product]:
            self.data[product][indicator_name] = []  # Initialize as list if missing

        if value is not None:
            self.data[product][indicator_name].append(value)  # Append new value

    def sort_buy_orders(self, buy_orders):
        return {int(price): int(volume) for price, volume in sorted(buy_orders.items(), key=lambda x: -int(x[0]))}

    def sort_sell_orders(self, sell_orders):    
        return {int(price): int(volume) for price, volume in sorted(sell_orders.items(), key=lambda x: int(x[0]))}

    def get_sell_order(self, sell_orders, rank=0):
        sell_orders = self.sort_sell_orders(sell_orders)
        if len(sell_orders.keys())>= rank+1:
            price, volume = list(sell_orders.items())[rank]
        else:
            price, volume = None, None
        return price, volume

    def get_buy_order(self, buy_orders, rank=0):
        buy_orders = self.sort_buy_orders(buy_orders)
        if len(buy_orders.keys())>= rank+1:
            price, volume = list(buy_orders.items())[rank]
        else:
            price, volume = None, None
        return price, volume

    def get_data_as_json(self) -> str:
        return json.dumps(self.data)

def is_available(best, best_amount):
    return best is not None and best_amount is not None

def adjust_sell_quantity(proposed_sell_quantity, max_sell_limit, current_position):
    if max_sell_limit >= 0 or proposed_sell_quantity>=0:
        raise ValueError(
            f"{proposed_sell_quantity=} or {max_sell_limit=},is supposed to be a negative number indicating sell")

    max_allowed_sell_quantity = max_sell_limit - current_position

    if proposed_sell_quantity < max_allowed_sell_quantity:
        adjusted_sell_quantity = max(proposed_sell_quantity, max_allowed_sell_quantity)
        remaining_sell_capacity = 0
    else:
        adjusted_sell_quantity = proposed_sell_quantity
        remaining_sell_capacity = max_allowed_sell_quantity - proposed_sell_quantity

    return adjusted_sell_quantity, remaining_sell_capacity

def adjust_buy_quantity(proposed_buy_quantity, max_buy_limit, current_position):
    if max_buy_limit <= 0 or proposed_buy_quantity<=0:
        raise ValueError(
            f"{proposed_buy_quantity=} or {max_buy_limit=},is supposed to be a positive number indicating buy")

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

def sma(prices, window):
    if len(prices) == 0 or window <= 0:
        raise ValueError("Prices list is empty or window size is invalid")
    
    if len(prices) ==1:
        return prices[0]
    
    # if window is larger than the list, adjust window length
    window = window if window < len(prices) else len(prices)

    # get the price subset
    price_subset = prices[-window:]  # get the last 'window' number of prices

    # calculate 
    sma = sum(price_subset)/window

    # print(f"{prices=} -- {window=}  --{price_subset=} -- {sma=}")
    return sma

def sigmoid(x, alpha):
    # alpha is a steepness parameter 2 is steep and 0.5
    return 1 / (1 + np.exp(-alpha * x))

def calculate_trend_score(current_price, sma_small, alpha):

    price_diff_small = current_price - sma_small
    
    # Apply the sigmoid function and scale to map to [-1, 1]
    trend_score = 2*(sigmoid(price_diff_small, alpha) - 0.5)

    return trend_score

def sma_crossover_score(td, product, prices, window_small, window_large, sigmoid_alpha):
    # returns a trend score between [-1, 1]
    # if there is a crossover, it returns [0, 1] for bullish and [0, -1] for bearish
    # depending on the price action associated with the crossover
    if len(prices) == 0 or window_small <= 0 or window_small <= 0:
        raise ValueError("Prices list is empty or window size is invalid")

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
        td.apply_indicator(product, "trend_score", 0)
        return 0

    # calculate current step    
    current_price = prices[-1]
    current_sma_small = sma(prices, window_small)
    current_sma_large = sma(prices, window_large)

    # update the sma indicators
    td.apply_indicator(product, sma_small_name, current_sma_small)
    td.apply_indicator(product, sma_large_name, current_sma_large)
    if len(prices) > window_large*.8:
        # from .8*the large window we start to make predictions
        td.apply_indicator(product, "trend_score", 0)
        return 0
    
    if current_sma_small > current_sma_large and previous_sma_small <= previous_sma_large:
        # bullish cross return value between [0, 1] 
        trend_score = calculate_trend_score(current_price, current_sma_small, sigmoid_alpha)
        trend_score = trend_score if trend_score > 0 else 0
        td.apply_indicator(product, "trend_score", current_sma_small)

    elif current_sma_small < current_sma_large and previous_sma_small >= previous_sma_large:
        # bearish cross return value between [0, 1] 
        trend_score = calculate_trend_score(current_price, current_sma_small, sigmoid_alpha)
        trend_score = trend_score if trend_score < 0 else 0
        td.apply_indicator(product, "trend_score", current_sma_small)
    else:
        trend_score = 0
        td.apply_indicator(product, "trend_score", current_sma_small)

    return trend_score

def order_imbalance_score(td, total_ask_volume, total_bid_volume):
    # Score ranges from -1 (strong selling) to +1 (strong buying).
    # 0 means neutral
    # strong positive (tighter spreads or higher bid prices)
    # strong negative (widen spreads or lower ask prices)
    if total_ask_volume is None or total_bid_volume is None:
        td.apply_indicator("imbalance_score", 0)        
        return 0

    nominator = abs(total_bid_volume) - abs(total_ask_volume)
    denominator = abs(total_bid_volume) + abs(total_ask_volume)

    if denominator == 0:
        td.apply_indicator("imbalance_score", 0)    
        return 0
    
    imbalance_score = nominator / denominator
    td.apply_indicator("imbalance_score", imbalance_score)

    return imbalance_score

def spread_to_price_ratio(td, best_ask, best_bid, mid_price):
    # Helps differentiate between stable vs. volatile products.
    # A higher ratio means a less liquid market → requires wider spreads.
    # A lower ratio suggests a liquid, competitive market.
    if best_ask is None or best_bid is None or mid_price is None:
        td.apply_indicator("liquidity_ratio", 0)    
        return 0

    spread = best_ask - best_bid
    ratio = spread / mid_price if mid_price != 0 else 0
    td.apply_indicator("liquidity_ratio", ratio)    

    return ratio

def calculate_dynamic_spread(td, product, spread_ratio):
    pass

def market_maker_strategy(td, product, orders):

    # get data all mid prices
    mid_prices = td.get_field(product, "mid_price")

    # 1. compute fair price
    fair_price = sma(mid_prices, 5)

    # 2. Identify market trend returns [-1 and 1]
    trend_score = sma_crossover_score(td, product, mid_prices, window_small=5, window_large=40, sigmoid_alpha=0.4)

    # 3. Analyze market pressure
    total_ask_volume = td.get_last_field(product, "total_ask_volume")
    total_bid_volume = td.get_last_field(product, "total_bid_volume")
    pressure_score = order_imbalance_score(td, total_ask_volume, total_bid_volume)

    # 4 liquidity
    best_bid = td.get_last_field(product, "best_bid")
    best_ask = td.get_last_field(product, "best_ask")
    liquidity_ratio = spread_to_price_ratio(td, best_ask, best_bid, mid_prices[-1])

    # 4. calculate dynamic Spread based on market conditions

    # 5. Define buy and sell conditions

    # 6 get orders
    
    return orders


class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}

        td = TradingData(state, position_limits)

        for product in state.order_depths:
            orders: List[Order] = []

            if product == "KELP":
                orders = market_maker_strategy(td, product, 5, orders)

            if product == "RAINFOREST_RESIN":
                orders = market_maker_strategy(td, product, 10, orders)
            
            result[product] = orders
        
        traderData = td.get_data_as_json()

        return result, conversions, traderData

# from mock import state, state2
if __name__ == "__main__":

    # arrs = [[],[1,2,3,6,6,6],[1],[None],[0]]
    # for prices in arrs:
    #     print(sma(prices, 0))

        # print(prices)
#     trader = Trader()
#     result,conversions, data = trader.run(state)
#     print(data)
#     print("--- NEW STEP ---") 
#     state2.traderData = data
#     result,conversions, data = trader.run(state2)
#     print(data)

    small = [1000, 1000, 1000, 1000, 1000]
    price = [996, 998, 1000, 1001, 1007]
    asks = [-10,-20,-30,-30,-30,-30]
    bids = [30,20,10,33,1,0]
    for a,b in zip(asks, bids):
        print(order_imbalance_score(10*a,10*b))