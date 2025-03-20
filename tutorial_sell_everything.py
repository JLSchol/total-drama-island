from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

def sell_all(order_depth, product, orders):

    lowest_bid_price, _ = min(order_depth.buy_orders.items(), key=lambda x: x[0])

    total_amount_bids = 0
    for vol in order_depth.buy_orders.values():
        total_amount_bids += vol

    sell_amount = total_amount_bids *-1
    print(f"SELL {product}: for {lowest_bid_price}$ -- with size: {sell_amount}")
    orders.append(Order(product, lowest_bid_price, sell_amount))
    return orders

class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = "string"

        print(state.position)

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if len(order_depth.sell_orders) != 0: # sell order exists
                orders = sell_all(order_depth, product, orders)

            if len(order_depth.buy_orders) != 0: # buy order exist
                pass

            result[product] = orders

        return result, conversions, traderData
