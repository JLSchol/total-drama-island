from datamodel import TradingState, Listing, OrderDepth, Trade, Observation, Order
from typing import List

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

observation = Observation( 
    plainValueObservations={"PRODUCT1": 6},
    conversionObservations={"PRODUCT2": 2}
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

def write_mock_to_json(filename):
    with open(filename + '.json', 'w') as f:
        f.write(state.toJSON())

def load_mock_tradingstate(filename):

if __name__ == '__main__':
    # write_mock_to_json("mock_trading_state")

    # analyze what to buy and/or sell based on outstanding buy and sell orders
        # output order

    # check position limit?

    for product in state.order_depths:
        print(product) #product names: PRODUCT1, PRODUCT2
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        result = {}

        if len(order_depth.sell_orders) != 0: # sell order exists
            best_ask, best_ask_amount = min(order_depth.sell_orders.items(), key=lambda x: x[0])
            if (best_ask < 10):
                orders.append(Order(product, best_ask, -1*best_ask_amount)) # append buy order

        if len(order_depth.buy_orders) != 0: # buy order exist
            best_bid, best_bid_amount = max(order_depth.buy_orders.items(), key=lambda x: x[0])
            if int(best_bid) > 10:
                # check position limit?
                orders.append(Order(product, best_bid, -1*best_bid_amount)) # append sell order

        result[product] = orders

    traderData = "string value holding trader state data required for nex iteration"
    conversions = -1 #?


    