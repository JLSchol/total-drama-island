from datamodel import TradingState, Listing, OrderDepth, Trade, Observation, Order
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
order_depth1 = OrderDepth(
)
order_depth1.buy_orders = {10: 7, 9: 5}
order_depth1.sell_orders= {11: -4, 12: -8}

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

if __name__ == '__main__':
    write_mock_to_json("mock_trading_state")