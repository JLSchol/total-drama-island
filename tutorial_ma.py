from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

class SMA:
    def __init__(self, price_list: List, length: int):
        self.price_list = price_list
        self.length = length

    def compute_ma(self) -> float:
        if len(self.price_list) == 0:
            return None  # No default price specified
        return sum(self.price_list) / len(self.price_list)
    
    def update_ma(self, price: float):
        self.price_list.append(price)
        if len(self.price_list) > self.length:
            self.price_list.pop(0)  # Maintain fixed size

    def get_sma(self, price: float):
        self.update_ma(price)
        return self.compute_ma()

class Indicators:
    def __init__(self, order_depth):
        self.order_depth = order_depth
        self.best_bid, self.best_bid_amount = self.get_best_bid(order_depth)
        self.best_ask, self.best_ask_amount = self.get_best_ask(order_depth)
        self.best_mid_price = self.get_best_mid_price(self.best_bid, self.best_ask, order_depth)
        self.best_weigthed_mid_price = self.get_best_weigthed_mid_price(self.best_bid, 
                                                                        self.best_bid_amount,
                                                                        self.best_ask, 
                                                                        self.best_ask_amount)
    
    def get_best_ask(self, order_depth):
        return min(order_depth.sell_orders.items(), key=lambda x: x[0]) if order_depth.sell_orders else (None, None)

    def get_best_bid(self, order_depth):
        return max(order_depth.buy_orders.items(), key=lambda x: x[0]) if order_depth.buy_orders else (None, None)

    def get_best_mid_price(self, best_bid, best_ask, order_depth):
        if best_bid is None or best_ask is None:
            return None
        return (self.get_best_bid(order_depth)[0] + self.get_best_ask(order_depth)[0]) / 2
        
    def get_best_weigthed_mid_price(self, best_bid, best_bid_amount, best_ask, best_ask_amount):
        if best_bid is None or best_ask is None:
            return None
        if best_bid_amount is None or best_ask_amount is None: 
            return None
        best_ask_amount = abs(best_ask_amount) # ask/sell quantities are denoted by (-) values
        return ( (best_bid*best_bid_amount) + (best_ask*best_ask_amount)) / (best_bid_amount+best_ask_amount)

def is_avaiable(best, best_amount):
    # Step 1: Check if the best bid is available
    if best is None or best_amount is None:
        return False  # Early return if the conditions are not met
    return True

def adjust_sell_quantity(proposed_sell_quantity, max_sell_limit, current_position):

    max_allowed_sell_quantity = max_sell_limit - current_position

    if proposed_sell_quantity < max_allowed_sell_quantity:
        adjusted_sell_quantity = max(proposed_sell_quantity, max_allowed_sell_quantity)
        remaining_sell_capacity = 0
    else:
        adjusted_sell_quantity = proposed_sell_quantity
        remaining_sell_capacity = max_allowed_sell_quantity - proposed_sell_quantity

    return adjusted_sell_quantity, remaining_sell_capacity

def adjust_buy_quantity(proposed_buy_quantity, max_buy_limit, current_position):

    max_allowed_buy_quantity = max_buy_limit - current_position

    if proposed_buy_quantity > max_allowed_buy_quantity:
        adjusted_buy_quantity = min(proposed_buy_quantity, max_allowed_buy_quantity)
        remaining_buy_capacity = 0
    else:
        adjusted_buy_quantity = proposed_buy_quantity
        remaining_buy_capacity = max_allowed_buy_quantity - proposed_buy_quantity

    return adjusted_buy_quantity, remaining_buy_capacity

def get_best_ask_buy_order(product, best_ask, best_ask_amount, current_position, max_position, orders) -> List[Order]:
    # Step 1: Check if the best ask is available
    if not is_avaiable(best_ask, best_ask_amount):
        return orders

    # Step 2: Calculate the buy quantity based on the best ask amount - flip signs: sell/ask is (-) and buy/bid is (+)
    buy_quantity = -1*best_ask_amount

    # Step 3: potentially limit buy_quantity based on current position
    buy_quantity, remaining_buy_capacity = adjust_buy_quantity(buy_quantity, max_position, current_position)

    if buy_quantity <= 0: # 0 = no order, and - numbers are sells
        return orders

    # step 4: append order to list of orders
    order = Order(product, best_ask, buy_quantity)
    print(f"sell: {order}")
    orders.append(Order(product, best_ask, buy_quantity))

    return orders

def get_best_bid_sell_order(product, best_bid, best_bid_amount, current_position, max_position, orders) -> List[Order]:
    # Step 1: Check if the best bid is available
    if not is_avaiable(best_bid, best_bid_amount):
        return orders

    # Step 2: Calculate the sell quantity based on the best bid amount
    sell_quantity = -1*best_bid_amount

    # Step 3: potentially limit sell_quantity based on current position
    max_sell_limit = -1*max_position # max position is given by an positive number so flip sign
    sell_quantity, remaining_sell_capacity = adjust_sell_quantity(sell_quantity, max_sell_limit, current_position)

    if sell_quantity >= 0: # 0 = no order, and + numbers are buys
        return orders

    # step 4: append order to list of orders
    order = Order(product, best_bid, sell_quantity)
    print(f"buy: {order}")
    orders.append(order)

    return orders

def sma_midprice_strategy(product, price_list, sma_length, ind, current_position, max_position, orders) -> List[Order]:
    """
    This is the strategy for buying and selling a product based of 
    the buying/selling if the ask/bid are below/above a fair price, which is calcuated by a sma of the best mid price
        1. calculate fair price based on moving average of the mid prace. This is the perceived true value of the product
        2. create buy order that:
            - tries to matches the lowest ask price and quantity
            - if the ask price is cheaper than the fair price -> $$$
        3. create sell order that:
            - tries to match highest bid price and quantity
            - if the bid price is more expensive than the fair price -> $$$
    """
    # calc fair price using sma with mid price. next update state.trader_data
    # print(f"OLD LIST: {price_list[product]}")
    sma = SMA(price_list[product], sma_length) # need to retrieve since data between runs is not persistent
    fair_price = sma.get_sma(ind.best_mid_price) # calc fair price based on sma of best_mid_price
    price_list[product] = sma.price_list # update the state.trader_data["price_list"][product] for next iteration
    # print(f"NEW LIST: {sma.price_list}")

    if fair_price is None:
        # print(f"fair_price is {fair_price} for {product}")
        return  orders # Skip trading if no price data available
    
    # if the best/lowest ask is less what we find fair, then 
    # try to only buy the best ask price and associated quantity
    if ind.best_ask < fair_price: 
        print(f"best_ask is {ind.best_ask} < fair price {fair_price} on product {product}")
        orders = get_best_ask_buy_order(product, 
                                    ind.best_ask, ind.best_ask_amount, 
                                    current_position, max_position, 
                                    orders)
    
    # if the best/highest bid is more than what we find fair, then 
    # try to only sell the best bid price and associated quantity
    if ind.best_bid > fair_price: 
        print(f"best_bid is {ind.best_bid} < fair price {fair_price} on product {product}")
        orders = get_best_bid_sell_order(product, 
                                    ind.best_bid, ind.best_bid_amount, 
                                    current_position, max_position, 
                                    orders)
    
    return orders

def get_kelp_orders(product, price_list, ind, current_position, max_position, orders) -> List[Order]:
    return sma_midprice_strategy(product, price_list, 5, ind, current_position, max_position, orders)

def get_rainforest_resin_orders(product, price_list, ind, current_position, max_position, orders) -> List[Order]:
    return sma_midprice_strategy(product, price_list, 10, ind, current_position, max_position, orders)



class Trader:
    
    def run(self, state: TradingState):
        result = {}
        conversions = 0
                
        # Load past prices from traderData
        try:
            trader_data = json.loads(state.traderData) if state.traderData else {"price_list": {}}
        except json.JSONDecodeError:
            trader_data = {"price_list": {}}
        
        # get old prices list from state.trader_data
        price_list = trader_data.get("price_list", {}) # if pastprices not exists, create empty dict
        
        position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}
        for product in state.order_depths:
            if product not in price_list: # if product not exsits in price_list, add key in dictionairy with empty list
                price_list[product] = []
            
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            ind = Indicators(order_depth)

            # Update with new mid prices
            if product == "KELP":
                orders = get_kelp_orders(product, price_list, ind,
                                         state.position.get(product, 0), position_limits[product],
                                         orders,
                                         )

            if product == "RAINFOREST_RESIN":
                orders = get_rainforest_resin_orders(product, price_list, ind,
                                         state.position.get(product, 0), position_limits[product],
                                         orders,
                                         )
            
            result[product] = orders
        
        # Store past prices in traderData for the next execution
        traderData = json.dumps(trader_data)
        
        return result, conversions, traderData
    
# from mock import state
# if __name__ == "__main__":
#     trader = Trader()
#     trader.run(state)
    