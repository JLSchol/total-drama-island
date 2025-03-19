import json
from datamodel import TradingState, Listing, OrderDepth, Trade, Observation, Order, ConversionObservation
from typing import List
import os

""" example
For the following example we assume a situation with two products:
1. PRODUCT1 with position limit 10
2. PRODUCT2 with position limit 20

At the start of the first iteration the run method is called with the TradingState

"""
# 2 listings (per product)
listing1 = Listing(
    symbol="PRODUCT1", # symbol
    product="PRODUCT1", # product
    denomination="SEASHELLS" # product
)
listing2 = Listing(
    symbol="PRODUCT2", # symbol
    product="PRODUCT2", # product
    denomination="SEASHELLS" # product
)

# 2 order depts (per product)
# keys are the price, and the values are quantities. 
# in buy orders the quantities are positive and in sell orders the quantity is negative
order_depth1 = OrderDepth(
)
order_depth1.buy_orders = {10: 7, 9: 5}
order_depth1.sell_orders= {13: -4, 12: -8} #of product 1, 11 for -4 and 12 for -8

order_depth2 = OrderDepth(
)
order_depth2.buy_orders = {142: 3, 141: 5}
order_depth2.sell_orders= {144: -5, 145: -8}

trade1 = Trade(
    symbol="PRODUCT1",
    price=11,
    quantity=4,
    buyer="buyerid",
    seller="sellerid",
    timestamp="900"
)
trade2 = Trade(
    symbol="PRODUCT2",
    price=8,
    quantity=1,
    buyer="buyerid",
    seller="sellerid",
    timestamp="1"
)

conversion_observation1 = ConversionObservation(
    bidPrice=120,
    askPrice=100,
    transportFees=0.1,
    exportTariff=0.2,
    importTariff=0.3,
    sugarPrice=1,
    sunlightIndex=10
)

conversion_observation2 = ConversionObservation(
    bidPrice=220,
    askPrice=200,
    transportFees=1,
    exportTariff=2,
    importTariff=3,
    sugarPrice=10,
    sunlightIndex=50
)

observation = Observation( 
    plainValueObservations={"PRODUCT1": 11,
                            "PRODUCT2":8},
    conversionObservations={"PRODUCT1": conversion_observation1,
                            "PRODUCT2": conversion_observation2}
)

state = TradingState(
    traderData="",
    timestamp=0,
    listings={"PRODUCT1": listing1,
              "PRODUCT2": listing2},
    order_depths= {"PRODUCT1": order_depth1,
                   "PRODUCT2": order_depth2},
    own_trades={"PRODUCT1": [],
                "PRODUCT2": []},
    market_trades={"PRODUCT1": [trade1],
                   "PRODUCT2": [trade2]},
    position={"PRODUCT1": 3,
              "PRODUCT2": -5},
    observations= observation
)
def get_mock_file_dir():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Go up to root/
    return os.path.join(root_dir, "mockfiles") # go down to mockfiles

def write_mock_to_json(file_name):
    file_path = os.path.join(get_mock_file_dir(), file_name)
    with open(file_path, 'w') as f:
        f.write(state.toJSON())

def load_tradingstate_json(file_name):
    file_path = os.path.join(get_mock_file_dir(), file_name)
    with open(file_path, "r") as file:
     return json.load(file)

if __name__ == '__main__':
    # # write to mockfiles dir
    # write_mock_to_json("mock_trading_state.json") 

    # # load a mockfile
    # tradingstate = load_tradingstate_json("mock_trading_state.json") 



    # analyze what to buy and/or sell based on outstanding buy and sell orders
        # output order

    # check position limit?
    result = {}
    for product in state.order_depths:
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []


        if len(order_depth.sell_orders) != 0: # sell order exists
            best_ask, best_ask_amount = min(order_depth.sell_orders.items(), key=lambda x: x[0])
            print(f"BUY {product}: for {best_ask}$ -- with size: {-1*best_ask_amount}")
            orders.append(Order(product, best_ask, -1*best_ask_amount)) # append buy order

        if len(order_depth.buy_orders) != 0: # buy order exist
            best_bid, best_bid_amount = max(order_depth.buy_orders.items(), key=lambda x: x[0])
            print(f"SELL {product}: for {best_bid}$ -- with size: {-1*best_bid_amount}")
            orders.append(Order(product, best_bid, -1*best_bid_amount)) # append sell order

        result[product] = orders
        print(result)
        print("-")

    traderData = "string value holding trader state data required for nex iteration"
    conversions = -1 #?
    print("---")
    print(result)

    