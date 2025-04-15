from datamodel import TradingState, Order, Trade
from typing import Dict, List, Optional, Tuple, Callable
import json
import numpy as np

class TradingData:
    def __init__(self, state: TradingState, position_limits: Dict[str, int], max_order_count: int, max_history_len: int):
        self.position_limits = position_limits
        self.max_order_count = max_order_count
        self.max_history_len = max_history_len
        self.data = self._initialize_data(state, position_limits)

    # Helper function to serialize numpy arrays
    def _numpy_array_default(self, obj):
        """Custom default function for serializing numpy arrays (including 2D arrays)."""
        if isinstance(obj, np.ndarray):
            # For 2D arrays, ensure it's serialized as a list of lists
            return obj.tolist()  # Convert 1D arrays to lists
        raise TypeError(f"Type {type(obj)} not serializable")

    def _convert_lists_to_numpy(self, data: Dict[str, Dict]):
        """Convert lists in JSON back to numpy arrays (including 2D arrays)."""
        for product, product_data in data.items():
            for key, value in product_data.items():
                if isinstance(value, list):
                    # Convert lists back to numpy arrays
                    arr = np.array(value, dtype=np.float64)
                    data[product][key] = arr
        return data

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
                    return self._update_new_state(self._convert_lists_to_numpy(data), state, position_limits)
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
            max_buy_position = int(position_limits[product])
            max_sell_position = int(-max_buy_position)

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
        sorted_prices = prices[sorted_indices][:max_order_count]
        sorted_volumes = volumes[sorted_indices][:max_order_count]

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
        sorted_prices = prices[sorted_indices][:max_order_count]
        sorted_volumes = volumes[sorted_indices][:max_order_count]

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

        if values is not None and values.size > 0:  # Check if values is a non-empty array
            # Check if the 2D array already has data, then append the new row
            if self.data[product][indicator_name].size == 0:
                # If the array is empty, initialize it with the new values as a single row
                self.data[product][indicator_name] = np.array([values])
            else:
                # If the array has data, append the new values as a new row
                self.data[product][indicator_name] = np.vstack([self.data[product][indicator_name], values])
                self.data[product][indicator_name] = self._truncate_array(self.data[product][indicator_name])

    def apply_3d_indicator(self, product: str, indicator_name: str, values: np.ndarray):
        if product not in self.data:
            self.data[product] = {}

        if indicator_name not in self.data[product]:
            self.data[product][indicator_name] = np.array([])

        if values is not None:
            # Check if the 3D array already has data, then append the new slice (2D array)
            if self.data[product][indicator_name].size == 0:
                # If the array is empty, initialize it with the new values as a single slice (2D array)
                self.data[product][indicator_name] = np.array([values])
            else:
                # If the array has data, append the new slice as a new entry along the first axis
                self.data[product][indicator_name] = np.concatenate([self.data[product][indicator_name], [values]], axis=0)
                self.data[product][indicator_name] = self._truncate_array(self.data[product][indicator_name])

    def get_last_field_3d_indicator(self, product: str, field: str) -> Optional[np.ndarray]:
        values = self.get_field(product, field)
        if values is None:
            return None
        if values.ndim == 3:
            return values[-1]  # Return the last slice (2D array) from the 3D array
        else:
            raise ValueError(f"Field {field} is not a 3D array but of type {values.ndim}-dimensional.")
    
    def get_data_as_json(self) -> str:
        # Serialize the data to JSON
        return json.dumps(self.data, default=self._numpy_array_default)

    def print_sim_step_data(self, print_list: list[str]):
        data = {}
        for product in self.data.keys():
            product_data = self.get_latest_fields(product)
            # filter product data based on print list
            filtered = {k: product_data[k] for k in product_data.keys() & print_list}
            data[product] = filtered
        # convert to json string, such that it can be loaded easily later
        print(json.dumps(data, default=self._numpy_array_default))

class OrderHelper:
    def __init__(self, td: TradingData):
        self.td = td

    def _is_available(self, best: Optional[float], best_amount: Optional[int]) -> bool:
        return best is not None and best_amount is not None

    def _adjust_sell_quantity(self, proposed_sell_quantity: int, max_sell_limit: int, current_position: int) -> Tuple[int, int]:
        if max_sell_limit >= 0 or proposed_sell_quantity >= 0:
            raise ValueError(f"{proposed_sell_quantity=} or {max_sell_limit=}, should be negative for selling")
        
        max_allowed_sell_quantity = max_sell_limit - current_position
        if proposed_sell_quantity < max_allowed_sell_quantity:
            adjusted_sell_quantity = max(proposed_sell_quantity, max_allowed_sell_quantity)
            new_position = max_sell_limit
        else:
            adjusted_sell_quantity = proposed_sell_quantity
            new_position = current_position + proposed_sell_quantity

        return adjusted_sell_quantity, new_position

    def _adjust_buy_quantity(self, proposed_buy_quantity: int, max_buy_limit: int, current_position: int) -> Tuple[int, int]:
        if max_buy_limit <= 0 or proposed_buy_quantity <= 0:
            raise ValueError(f"{proposed_buy_quantity=} or {max_buy_limit=}, should be positive for buying")
        
        max_allowed_buy_quantity = max_buy_limit - current_position
        if proposed_buy_quantity > max_allowed_buy_quantity:
            adjusted_buy_quantity = min(proposed_buy_quantity, max_allowed_buy_quantity)
            new_position = max_buy_limit
        else:
            adjusted_buy_quantity = proposed_buy_quantity
            new_position = current_position + proposed_buy_quantity
        return adjusted_buy_quantity, new_position
    
    def _adjust_quantity(self, order_type: str, product: str, proposed_buy_quantity: int, max_buy_limit: int, current_position: int) -> Tuple[int, int]:
        
        try:
            if order_type == "buy":
                order_quantity, new_position = self._adjust_buy_quantity(proposed_buy_quantity, max_buy_limit, current_position)
                if order_quantity <= 0:
                    print(f"Skipped 0-qty or invalid buy: {product=} {order_type=} {current_position=} {max_buy_limit=}")
                    return 0, current_position
            elif order_type == "sell":
                order_quantity, new_position = self._adjust_sell_quantity(proposed_buy_quantity, max_buy_limit, current_position)
                if order_quantity >= 0:
                    print(f"Skipped 0-qty or invalid sell: {product=} {order_type=} {current_position=} {max_buy_limit=}")
                    return 0, current_position
            else:
                raise ValueError(f"Invalid order type: {order_type}")
        except ValueError as e:
            print(f"Adjustment failed for {product=} {order_type=}: {e}")
            return 0, current_position

        return order_quantity, new_position

    def _get_best_order(self, order_type: str, product: str, price: int, amount: int, current_position: int, max_position: int, orders: List[Order]) -> Tuple[List[Order], int]:
        
        if not self._is_available(price, amount):
            print(f"Invalid price or amount for {product=}: {price=}, {amount=}")
            return orders, None  # No valid price or quantity available

        # Adjust based on position limits
        order_quantity, new_position = self._adjust_quantity(order_type, product, amount, max_position, current_position)

        if order_quantity == 0:
            return orders, new_position

        # Append the order
        orders.append(Order(product, int(price), int(order_quantity)))
        return orders, new_position

    def get_order(self, order_type: str, product: str, price: int, amount: int, current_position:int, max_position: int):
        
        if not self._is_available(price, amount):
            print(f"invalid order: {product=}: {price=}, {amount=}")
            return None, None  # No valid price or quantity available
        
        # Adjust based on position limits
        order_quantity, new_position = self._adjust_quantity(order_type, product, amount, max_position, current_position)

        if order_quantity == 0:
            return None, new_position
        
        return Order(product, int(price), int(order_quantity)), new_position

    def get_best_order(self, order_type: str, product: str, orders: List[Order]) -> Tuple[List[Order], int]:

        if order_type == "buy":
            price = self.td.get_last_field(product, "best_ask")
            amount = self.td.get_last_field(product, "best_ask_volume")
            max_position = self.td.get_last_field(product, "max_buy_position")
        elif order_type == "sell":
            price = self.td.get_last_field(product, "best_bid")
            amount = self.td.get_last_field(product, "best_bid_volume")
            max_position = self.td.get_last_field(product, "max_sell_position")
        else:
            raise ValueError(f"first argument should be 'buy' or 'sell' and is {order_type=}")

        current_position = self.td.get_last_field(product, "current_position")

        orders, new_position = self._get_best_order(order_type, product, price, -amount, current_position, max_position, orders)
        return orders, new_position
        
    def get_all_orders_given_conditions(self, order_type: str, product: str, orders: List[Order],
                        price_condition: Optional[Callable[[float], bool]] = None,
                        quantity_condition: Optional[Callable[[int], bool]] = None,
                        position_condition: Optional[Callable[[int], bool]] = None) -> Tuple[List[Order], int]:

        # Default conditions: accept everything
        price_condition = price_condition or (lambda price: True)
        quantity_condition = quantity_condition or (lambda amount: True)
        position_condition = position_condition or (lambda position: True)

        if order_type == "buy":
            prices = self.td.get_last_field(product, "ask_prices")
            amounts = self.td.get_last_field(product, "ask_volumes")
            max_position = self.td.get_last_field(product, "max_buy_position")

        elif order_type == "sell":
            prices = self.td.get_last_field(product, "bid_prices")
            amounts = self.td.get_last_field(product, "bid_volumes")
            max_position = self.td.get_last_field(product, "max_sell_position")
        else:
            raise ValueError(f"first argument should be 'buy' or 'sell' and is {order_type=}")

        position = self.td.get_last_field(product, "current_position")

        for price, amount in zip(prices,amounts):
            if price is None or (isinstance(price, float) and np.isnan(price)): # check for numpy and normal None
                return orders, position
            if amount is None or (isinstance(price, float) and np.isnan(amount)): # check for numpy and normal None
                return orders, position
            if position == max_position: # do not create order with quantity 0
                return orders, position
            
            # flip because we want to do the opposite in our order 
            # (ask quantity is negative, so if we buy, we need to flip sign to positive to indicate buy)
            amount = -amount 

            if price_condition(price) and quantity_condition(amount) and position_condition(position):
                order, position = self.get_order(order_type, product, price, amount, position, max_position)

                if order is None:
                    return orders, position

                orders.append(order)

        return orders, position

    def get_all_order(self, order_type: str, product: str, orders: List[Order]) -> Tuple[List[Order], int]:
        orders, position = self.get_all_orders_given_conditions(order_type, product, orders, None, None, None)
        return orders, position

    def get_all_buy_orders_below_fairprice(self, product: str, fairprice: float, orders: List[Order]) -> Tuple[List[Order], int]:
        orders, position = self.get_all_orders_given_conditions("buy", product, orders, 
                                                                price_condition=lambda price: price < fairprice,
                                                                quantity_condition=None,
                                                                position_condition=None)

def get_kalman_acceleration_params(dt: float = 1.0):

    # # Initial state: [price, velocity, acceleration]
    initial_state = np.array([0.0, 0.0, 0.0])
    initial_covariance = np.eye(3)

    # Process noise: [price, velocity, acceleration]
    process_noise = np.diag([1e-3, 1e-4, 1e-5])
    measurement_noise = np.array([[1e-2]])  # Measurement noise for price

    # Transition function
    def f(x):
        return np.array([
            x[0] + x[1]*dt + 0.5*x[2]*dt**2,
            x[1] + x[2]*dt,
            x[2]
        ])

    # Jacobian of f
    def F(x):
        return np.array([
            [1, dt, 0.5*dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])

    # Measurement function: observe price only
    def h(x):
        return np.array([x[0]])

    # Jacobian of h
    def H(x):
        return np.array([[1, 0, 0]])

    return initial_state, initial_covariance, process_noise, measurement_noise, f, F, h, H

def get_kalman_velocity_params(dt: float = 1.0):

    # Initial state: [price, velocity]
    initial_state = np.array([0.0, 0.0])
    initial_covariance = np.eye(2)

    # Process noise: [price, velocity]
    process_noise = np.diag([1e-3, 1e-4])
    measurement_noise = np.array([[1e-2]])  # Measurement noise for price

    # Transition function
    def f(x):
        return np.array([
            x[0] + x[1] * dt,
            x[1]
        ])

    # Jacobian of f
    def F(x):
        return np.array([
            [1, dt],
            [0, 1]
        ])

    # Measurement function: observe price only
    def h(x):
        return np.array([x[0]])

    # Jacobian of h
    def H(x):
        return np.array([[1, 0]])

    return initial_state, initial_covariance, process_noise, measurement_noise, f, F, h, H

def extended_kalman_filter_live_3D(prices: list[float], product: str, td: TradingData) -> float:

    # Get model
    # initial_state, initial_covariance, Q, R, f, F, h, H = get_kalman_acceleration_params()
    initial_state, initial_covariance, Q, R, f, F, h, H = get_kalman_velocity_params()

    # Restore previous state or init
    x = td.get_last_field(product, "kf_x")
    P = td.get_last_field_3d_indicator(product, "kf_P")

    if x is None or P is None:
        # print("non")
        x = initial_state.copy()
        x[0] = prices[0]  # Set initial price
        P = initial_covariance.copy()
    else:
        # print(x)
        x = np.array(x, dtype=np.float64).flatten()
        P = np.array(P, dtype=np.float64)

        # # Defensive check
        # if x.ndim != 1 or (x.shape[0] != 3 or x.shape[0] != 2):
        #     raise ValueError(f"Invalid state shape for x: {x.shape}, expected (3 or 2,) — got: {x}")


    # === Kalman update ===
    z = np.array([prices[-1]])  # latest price
    x_pred = f(x)
    F_jac = F(x)
    P_pred = F_jac @ P @ F_jac.T + Q

    H_jac = H(x_pred)
    y = z - h(x_pred)
    S = H_jac @ P_pred @ H_jac.T + R
    K = P_pred @ H_jac.T @ np.linalg.inv(S)

    x_upd = x_pred + K @ y
    P_upd = (np.eye(len(x)) - K @ H_jac) @ P_pred

    # === Store state ===
    td.apply_2d_indicator(product, "kf_x", x_upd)
    td.apply_3d_indicator(product, "kf_P", P_upd)

    # Return fair price
    return x_upd[0]

def linear_trend(prices, window=10):
    actual_window = min(len(prices), window)
    if actual_window < 2:
        return 0  # Still not enough data to form a trend

    y = np.array(prices[-actual_window:], dtype=np.float64)
    x = np.arange(actual_window)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0

    slope = numerator / denominator

    # Normalize the slope by dividing by the standard deviation of the y values
    std_dev = np.std(y) if np.std(y) > 0 else 1  # Avoid division by zero
    normalized_slope = slope / std_dev

    return normalized_slope

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

def sma_strategy(td: TradingData, oh: OrderHelper, product: str, orders: list[Order], 
                 price_type: str, sma_window: int):
    allowed_types = ["mid_price", "weighted_mid_price"]
    if price_type not in allowed_types:
        raise ValueError(f"price_type must be one of {allowed_types}")
    
    # 1. Get the latest price and SMA
    prices = td.get_field(product, price_type)
    # fair_price = sma(prices, sma_window)
    fair_price = extended_kalman_filter_live_3D(prices, product, td)
    
    td.apply_indicator(product, "fair_price", fair_price)

    # 2 get the orders
    best_ask = td.get_last_field(product, "best_ask")
    best_bid = td.get_last_field(product, "best_bid")

    if best_ask is not None and best_ask < fair_price:
        orders, _ = oh.get_best_order("buy", product, orders)

    if best_bid is not None and best_bid > fair_price:
        orders, _ = oh.get_best_order("sell", product, orders)

    return orders

def sma_strategy_get_all(td: TradingData, oh: OrderHelper, product: str, orders: list[Order], 
                 price_type: str, sma_window: int):
    allowed_types = ["mid_price", "weighted_mid_price"]
    if price_type not in allowed_types:
        raise ValueError(f"price_type must be one of {allowed_types}")
    
    # 1. Get the latest price and SMA
    prices = td.get_field(product, price_type)
    fair_price = sma(prices, sma_window)
    td.apply_indicator(product, "fair_price", fair_price)

    # get all buy orders of which 
    #   the ask price is below the fair price 
    #   check for all asks, (not only best ask)
    #   buy maximum available quantity for each.  
    orders, _ = oh.get_all_orders_given_conditions("buy", product, orders, 
                                                                price_condition=lambda price: price < fair_price,
                                                                quantity_condition=None,
                                                                position_condition=None)
    # same idea for sell
    orders, _ = oh.get_all_orders_given_conditions("sell", product, orders, 
                                                                price_condition=lambda price: price > fair_price,
                                                                quantity_condition=None,
                                                                position_condition=None)

    return orders

def squid_ink_strategy(td: TradingData, oh: OrderHelper, product: str, orders: list[Order],
                       price_type: str,
                       lookback, max_lookback) -> list[Order]:
    
    def calc_dynamic_momentum_diff(prices, lookback, max_lookback):
        if lookback <1:
            raise ValueError(f"{lookback=} and should be larger than 1")
        if lookback > max_lookback:
            raise ValueError(f"{lookback=} is not allowed to bigger than {max_lookback=}")

        if len(prices) == 1:
            return 0
        
        # if the array is not big enough to reach max lookback, return earlier with max length of array
        if len(prices)-1 <= lookback:  
            return prices[-1] - prices[0]
        
        # if the max lookback is reached recursively
        if lookback == max_lookback:
            return prices[-1] - prices[-(max_lookback+1)]

        # calculate a momentum metric (sort of difference) metric (we use this as +/- change to identify a recent peak.
        diff = prices[-1] - prices[-(lookback+1)]

        # if the difference is the same, it can be a flat line and we want to look further back
        # call recursevly
        if diff == 0:
            return calc_dynamic_momentum_diff(prices, lookback+1, max_lookback)

        return diff
    
    allowed_types = ["mid_price", "weighted_mid_price"]
    if price_type not in allowed_types:
        raise ValueError(f"price_type must be one of {allowed_types}")
    
    # 1. Get the latest price and SMA
    prices = td.get_field(product, price_type)
    fair_price = sma(prices, 3)
    td.apply_indicator(product, "fair_price", fair_price)

    prices = td.get_field(product, price_type)

    # 2. get previous momentum diff
    previous_momentum_diff = td.get_last_field(product,"momentum_diff")

    # 3. calculate dynamic momentum diff
    momentum_diff = calc_dynamic_momentum_diff(prices, lookback, max_lookback)
    td.apply_indicator(product, "momentum_diff", momentum_diff)
    
    # 4. create buy/sell signal by noticing momentum shift.
    if previous_momentum_diff is None:
        return orders
    
    if previous_momentum_diff<0 and momentum_diff>0:
        # down to up
        momentum_signal = 1
    elif previous_momentum_diff>0 and momentum_diff<0:
        # up to down
        momentum_signal = -1
    else:
        momentum_signal = 0

    td.apply_indicator(product, "momentum_signal", momentum_signal)

    # 2 get the orders
    best_ask = td.get_last_field(product, "best_ask")
    best_bid = td.get_last_field(product, "best_bid")


    # TODO: 
    # strat idea:
        # trade with momentum (e.g. difference of longer sma is positive)
        # if momentum is up, enter long position.
        # track buy orders and cost cost.
        # do not wait for momentum signal to flip, instead buy when sell price above buy order and try to close all positions before the longer sma switches
    
    # strat idea:
        # predict fair price based on slope sma (future price because of lagging signal)

    # strat idea:
        # something with tracking average buy/sell?

    if momentum_signal ==1 and best_ask < fair_price:
        # buy
        orders.append(Order(product, int(best_ask), int(1)))

    if momentum_signal ==-1 and best_bid > fair_price:
        # sell
        orders.append(Order(product, int(best_bid), int(-1)))

    return orders

def squid_ink_strategy2(td: TradingData, oh: OrderHelper, product: str, orders: list[Order],
                       price_type: str,
                       lookback, max_lookback) -> list[Order]:
    
    def calc_dynamic_momentum_diff(prices, lookback, max_lookback):
        if lookback <1:
            raise ValueError(f"{lookback=} and should be larger than 1")
        if lookback > max_lookback:
            raise ValueError(f"{lookback=} is not allowed to bigger than {max_lookback=}")

        if len(prices) == 1:
            return 0
        
        # if the array is not big enough to reach max lookback, return earlier with max length of array
        if len(prices)-1 <= lookback:  
            return prices[-1] - prices[0]
        
        # if the max lookback is reached recursively
        if lookback == max_lookback:
            return prices[-1] - prices[-(max_lookback+1)]

        # calculate a momentum metric (sort of difference) metric (we use this as +/- change to identify a recent peak.
        diff = prices[-1] - prices[-(lookback+1)]

        # if the difference is the same, it can be a flat line and we want to look further back
        # call recursevly
        if diff == 0:
            return calc_dynamic_momentum_diff(prices, lookback+1, max_lookback)

        return diff
    
    allowed_types = ["mid_price", "weighted_mid_price"]
    if price_type not in allowed_types:
        raise ValueError(f"price_type must be one of {allowed_types}")
    
    # 1. Get the latest price
    prices = td.get_field(product, price_type)
    
    # 2. calc fair price (extende kalman filter)
    fair_price = extended_kalman_filter_live_3D(prices, product, td)
    td.apply_indicator(product, "fair_price", fair_price)

    # 3. calc trend (linear regression)
    fair_prices = td.get_field(product, "fair_price")
    trend_strength_normalized  = linear_trend(fair_prices, window=10)
    td.apply_indicator(product, "trend_strength_normalized", trend_strength_normalized)

    if trend_strength_normalized > 0.1:
        trend = 1
    elif trend_strength_normalized < -0.1:
        trend = -1
    else:
        trend = 0

    # 4. signal if recent peak or valley (mom_shift_signal)
    previous_momentum_diff = td.get_last_field(product,"momentum_diff")
    momentum_diff = calc_dynamic_momentum_diff(prices, lookback, max_lookback)
    td.apply_indicator(product, "momentum_diff", momentum_diff)
    
    # 4. create buy/sell signal by noticing momentum shift.
    # (to know if previous timestep was a, peak or valley)
    if previous_momentum_diff is None:
        return orders
    
    if previous_momentum_diff<0 and momentum_diff>0:
        # down to up
        mom_shift_signal = 1
    elif previous_momentum_diff>0 and momentum_diff<0:
        # up to down
        mom_shift_signal = -1
    else:
        mom_shift_signal = 0

    td.apply_indicator(product, "mom_shift_signal", mom_shift_signal)

    # buy_averages = td.get_field(product, "buy_average")
    # buy_counts = td.get_field(product, "buy_count")
    # sell_averages = td.get_field(product, "sell_average")
    # sell_counts = td.get_field(product, "sell_count")

    
    # # Handle None values safely
    # buy_averages_l = min(len(buy_averages), 5) if buy_averages is not None else 0
    # sell_averages_l = min(len(sell_averages), 5) if sell_averages is not None else 0

    # # Compute safe buy_average
    # if (
    #     buy_averages_l == 0 or 
    #     buy_counts is None or 
    #     len(buy_counts) < buy_averages_l or 
    #     any(v is None for v in buy_averages[-buy_averages_l:]) or 
    #     any(c is None for c in buy_counts[-buy_averages_l:])
    # ):
    #     buy_average = None
    # else:
    #     buy_count_sum = sum(buy_counts[-buy_averages_l:])
    #     buy_average = sum(buy_averages[-buy_averages_l:]) / buy_count_sum if buy_count_sum != 0 else None

    # # Compute safe sell_average
    # if (
    #     sell_averages_l == 0 or 
    #     sell_counts is None or 
    #     len(sell_counts) < sell_averages_l or 
    #     any(v is None for v in sell_averages[-sell_averages_l:]) or 
    #     any(c is None for c in sell_counts[-sell_averages_l:])
    # ):
    #     sell_average = None
    # else:
    #     sell_count_sum = sum(sell_counts[-sell_averages_l:])
    #     sell_average = sum(sell_averages[-sell_averages_l:]) / sell_count_sum if sell_count_sum != 0 else None

    
    
    if trend == 1:
        # buy directly after dip based on mom_shift_signal
        if mom_shift_signal == 1:
            # buy all ask orders (trend is up so no restrictino on buying)     
            orders, position = oh.get_all_orders_given_conditions("buy", product, orders, 
                                                                    price_condition=lambda price: price < fair_price + 1,
                                                                    # price_condition=None,
                                                                    quantity_condition=None,
                                                                    position_condition=None)
            
            # sell if price is above fair_price
            # when selling, never let position be < 0 since the trend is going up
        
        if mom_shift_signal == -1:
            orders, position = oh.get_all_orders_given_conditions("sell", product, orders, 
                                                                    # price_condition=lambda price: price > buy_average if buy_average is not None else None,
                                                                    price_condition=lambda price: price > fair_price,
                                                                    quantity_condition=None,
                                                                    position_condition=lambda position: position >= 0)

    if trend == -1:
        # short directly after peak, based on mom_shift_signal (neglect fair_price condition?)
        if mom_shift_signal == -1:
            orders, position = oh.get_all_orders_given_conditions("sell", product, orders, 
                                                                    price_condition=lambda price: price > fair_price -1,
                                                                    # price_condition=None,
                                                                    quantity_condition=None,
                                                                    position_condition=None)
    
        # buy if price is above fair price
        # when buying, never let position be > 0 since the rend is going down
        if mom_shift_signal == 1:
            orders, position = oh.get_all_orders_given_conditions("buy", product, orders, 
                                                                    # price_condition=lambda price: price < sell_average if sell_average is not None else None,
                                                                    price_condition=lambda price: price < fair_price,
                                                                    quantity_condition=None,
                                                                    position_condition=lambda position: position <= 0)

    if trend == 0:
        # get the orders
        if mom_shift_signal == 1:
            # buy all ask orders thats below fair price        
            orders, position = oh.get_all_orders_given_conditions("buy", product, orders, 
                                                                    price_condition=lambda price: price < fair_price,
                                                                    quantity_condition=None,
                                                                    position_condition=None)

        # sell all bid orders thats below fair price      
        if mom_shift_signal == -1:
            orders, position = oh.get_all_orders_given_conditions("sell", product, orders, 
                                                                    price_condition=lambda price: price > fair_price,
                                                                    quantity_condition=None,
                                                                    position_condition=None)
    
    return orders

def track_trades(trades: Dict[str, List[Trade]], td: TradingData):
    # get this from state (one timestamp after it is submitted)
    
    def update_trade_stats(type: str, product: str, td: TradingData,
                           quantities: int, prices: int):
        if len(quantities) ==0 or len(prices)==0:
            return

        if type == "buy":
            id = "buy"
        
        elif type == "sell":
            id = "sell"
        else:
            raise ValueError("first arg {type=} is not correct, must be 'buy' or 'sell'")
    
        # define names based on buy/sell
        # total_count_name = f"total_{id}_count"
        # total_price_name = f"total_{id}_price"
        # total_average_name = f"total_{id}_average"
        count_name = f"{id}_count"
        price_name = f"{id}_price"
        average_name = f"{id}_average"

        # combine orders of different prices and quantities in one
        total_q, total_r, avarage = 0,0,0
        for q, p in zip(quantities, prices):
            total_q += q
            total_r += q*p
            avarage = total_r/total_q

        # calculate new values
        # total_count, total_price, total_average = calc_new_trade_stats(total_count, total_price, 
        #                                                             quantity,price)
        # update in class
        td.apply_indicator(product, count_name, total_q)
        td.apply_indicator(product, price_name, total_r)
        td.apply_indicator(product, average_name, avarage)


    for product, trades in trades.items():       
        quantities_bought, prices_bought = [], []
        quantities_sold, prices_sold = [], []
        for trade in trades: 
            # determine buy or sell
            if trade.buyer == "SUBMISSION":
                quantities_bought.append(trade.quantity)
                prices_bought.append(trade.price)
                

            if trade.seller == "SUBMISSION":
                quantities_sold.append(trade.quantity)
                prices_sold.append(trade.price)
        
        
        update_trade_stats("buy", product, td, quantities_bought, prices_bought)
        update_trade_stats("buy", product, td, quantities_sold, prices_sold)

class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        position_limits = { "RAINFOREST_RESIN": 50, "KELP": 50, 
                            "SQUID_INK": 50,
                            "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
                            "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
                            "VOLCANIC_ROCK": 400,
                            "VOLCANIC_ROCK_VOUCHER_9500": 200,
                            "VOLCANIC_ROCK_VOUCHER_9750": 200,
                            "VOLCANIC_ROCK_VOUCHER_10000": 200,
                            "VOLCANIC_ROCK_VOUCHER_10250": 200,
                            "VOLCANIC_ROCK_VOUCHER_10500": 200
                           }
        
        picic_content = {"PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                         "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2 }}
        
        volcanic_voucher = {
                        "VOLCANIC_ROCK_VOUCHER_9500": 9500,
                        "VOLCANIC_ROCK_VOUCHER_9750": 9750,
                        "VOLCANIC_ROCK_VOUCHER_10000": 10000,
                        "VOLCANIC_ROCK_VOUCHER_10250": 10250,
                        "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        
        max_order_count = 3 #  order book depth + 1 = 4 (need the +1 to be able to pop from list)
        max_history = 11

        td = TradingData(state, position_limits, max_order_count, max_history)
        oh = OrderHelper(td)


        track_trades(state.own_trades, td)

        for product in state.order_depths:
            orders: List[Order] = []
            # tutorial
            if product == "KELP":
                orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)
            if product == "RAINFOREST_RESIN":
                orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)

            # round 1
            if product == "SQUID_INK":
                orders = squid_ink_strategy2(td, oh, product, orders, "mid_price", 1,6)

                

            # round 2
            # if product == "CROISSANTS":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)
            # if product == "JAMS":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)
            # if product == "DJEMBES":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)
            # if product == "PICNIC_BASKET1":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)
            # if product == "PICNIC_BASKET2":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)

            # # round 3
            # if product == "VOLCANIC_ROCK":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)

            # if product == "VOLCANIC_ROCK_VOUCHER_9500":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)

            # if product == "VOLCANIC_ROCK_VOUCHER_9750":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)

            # if product == "VOLCANIC_ROCK_VOUCHER_10000":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)

            # if product == "VOLCANIC_ROCK_VOUCHER_10250":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)

            # if product == "VOLCANIC_ROCK_VOUCHER_10500":
            #     orders = sma_strategy_get_all(td, oh, product, orders, "mid_price", 5)


            
            result[product] = orders


        # td.print_sim_step_data(["timestamp", "bid_prices", "bid_volumes", "ask_prices", "ask_volumes", 
        #                 "mid_price", "current_position", 
        #                 "fair_price", "momentum_diff", "mom_shift_signal"
        #                 "total_buy_count", "total_buy_price", "buy_average",
        #                 "total_sell_count", "total_sell_price", "sell_average"])

        traderData = td.get_data_as_json()

        return result, conversions, traderData


# from mock import state, state2
# if __name__ == "__main__":


#     trader = Trader()
#     result,conversions, data = trader.run(state)
#     # print(data)
#     print()
#     print()
#     print()
#     print("--- NEW STEP ---") 
#     print()
#     print()
#     print()
#     state2.traderData = data
#     result,conversions, data = trader.run(state2)
#     print(data)


#     trader = Trader()
#     newstate = state
#     for i in range(3):
        
#         result,conversions, data = trader.run(newstate)
#         newstate.traderData = data


