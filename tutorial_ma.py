from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

class Trader:
   
    def compute_moving_average(self, past_prices):
        if len(past_prices) == 0:
            return None  # No default price specified
        return sum(past_prices) / len(past_prices)
    
    # Best Ask Price: The lowest price that sellers are willing to accept for an asset.
    def get_best_ask(self, order_depth):
        return min(order_depth.sell_orders.items(), key=lambda x: x[0]) if order_depth.sell_orders else (None, None)
    
    # Best Bid Price: The highest price that buyers are willing to pay for an asset.
    def get_best_bid(self, order_depth):
        return max(order_depth.buy_orders.items(), key=lambda x: x[0]) if order_depth.buy_orders else (None, None)
    
    def update_past_prices(self, product, best_bid, best_ask, past_prices, maxlen):
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2
            past_prices[product].append(mid_price)
            if len(past_prices[product]) > maxlen:
                past_prices[product].pop(0)  # Maintain fixed size
    
    def run(self, state: TradingState):
        result = {}
        position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        maxlen = 5  # Store last 5 prices for SMA
        
        # Load past prices from traderData
        try:
            trader_data = json.loads(state.traderData) if state.traderData else {"past_prices": {}}
        except json.JSONDecodeError:
            trader_data = {"past_prices": {}}
        
        past_prices = trader_data.get("past_prices", {})
        
        for product in state.order_depths:
            if product not in past_prices:
                past_prices[product] = []
            
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            # Get best bid and ask prices
            best_ask, best_ask_amount = self.get_best_ask(order_depth)
            best_bid, best_bid_amount = self.get_best_bid(order_depth)
            
            # Update past prices
            self.update_past_prices(product, best_bid, best_ask, past_prices, maxlen)
            
            # Compute acceptable price using moving average
            acceptable_price = self.compute_moving_average(past_prices[product])
            if acceptable_price is None:
                continue  # Skip trading if no price data available
            
            # Get current position
            current_position = state.position.get(product, 0)
            max_position = position_limits[product]
            
            # Buying logic (ensuring we don’t exceed max position)
            if best_ask is not None and best_ask_amount is not None:
                buy_quantity = min(-best_ask_amount, max_position - current_position)
                if best_ask < acceptable_price and buy_quantity > 0:
                    orders.append(Order(product, best_ask, buy_quantity))
            
            # Selling logic (ensuring we don’t exceed min position)
            if best_bid is not None and best_bid_amount is not None:
                sell_quantity = min(best_bid_amount, max_position + current_position)
                if best_bid > acceptable_price and sell_quantity > 0:
                    orders.append(Order(product, best_bid, -sell_quantity))
            
            result[product] = orders
        
        # Store past prices in traderData for the next execution
        traderData = json.dumps({"past_prices": past_prices})
        
        return result, 0, traderData
