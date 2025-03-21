import json
from datamodel import TradingState, Listing, OrderDepth, Trade, Observation, Order, ConversionObservation
from typing import List
import os

""" example
For the following example we assume a situation with two products:
1. RAINFOREST_RESIN with position limit 50
2. KELP with position limit 50

At the start of the first iteration the run method is called with the TradingState

"""
# 2 listings (per product)
listing1 = Listing(
    symbol="RAINFOREST_RESIN", # symbol
    product="RAINFOREST_RESIN", # product
    denomination="SEASHELLS" # product
)
listing2 = Listing(
    symbol="KELP", # symbol
    product="KELP", # product
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
    symbol="RAINFOREST_RESIN",
    price=11,
    quantity=4,
    buyer="buyerid",
    seller="sellerid",
    timestamp="900"
)
trade2 = Trade(
    symbol="KELP",
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
    plainValueObservations={"RAINFOREST_RESIN": 11,
                            "KELP":8},
    conversionObservations={"RAINFOREST_RESIN": conversion_observation1,
                            "KELP": conversion_observation2}
)

state = TradingState(
    traderData="",
    timestamp=0,
    listings={"RAINFOREST_RESIN": listing1,
              "KELP": listing2},
    order_depths= {"RAINFOREST_RESIN": order_depth1,
                   "KELP": order_depth2},
    own_trades={"RAINFOREST_RESIN": [],
                "KELP": []},
    market_trades={"RAINFOREST_RESIN": [trade1],
                   "KELP": [trade2]},
    position={"RAINFOREST_RESIN": 3,
              "KELP": -5},
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


def buy_all(order_depth, product, position_limits, position, orders):

    highest_ask_price, _ = max(order_depth.sell_orders.items(), key=lambda x: x[0])

    total_amount_asks = 0
    for vol in order_depth.sell_orders.values():
        total_amount_asks += vol

    buy_amount = total_amount_asks *-1

    print(product)
    print()
    print(position_limits[product])
    print()
    print(position[product])
    print()

    buy_room = position_limits[product] - position.get(product, 0)
    if buy_amount > buy_room:
        buy_amount = buy_room

    print(f"BUY {product}: for {highest_ask_price}$ -- with size: {buy_amount}")
    orders.append(Order(product, highest_ask_price, buy_amount))

    return orders

def get_state():
    return state

def run(state: TradingState):
        result = {}
        conversions = 0
        traderData = "string"
        position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50}  # Position limits

        print(f"{state.timestamp=}")
        print("\n")
        print(f"{state.own_trades=}")
        print("\n")
        print(f"{state.market_trades=}")
        print("\n")
        print(f"{state.position=}")

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if len(order_depth.sell_orders) != 0: # sell order exists
                orders = buy_all(order_depth, product, position_limits, state.position, orders)

            if len(order_depth.buy_orders) != 0: # buy order exist
                pass

            result[product] = orders

        return result, conversions, traderData


if __name__ == '__main__':
    # # write to mockfiles dir
    # write_mock_to_json("mock_trading_state.json") 

    # # load a mockfile
    # tradingstate = load_tradingstate_json("mock_trading_state.json") 



    result, conversions, traderData = run(state)

    print(result)

    