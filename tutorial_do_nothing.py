from datamodel import OrderDepth, TradingState, Order
from typing import List
import json


class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0
        traderData = "string"
        # prev_state = json.loads(state.traderData)

        # print(f"prev state position: {prev_state.position}\ncurr state position: {state.position}")

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if len(order_depth.sell_orders) != 0: # sell order exists
                pass

            if len(order_depth.buy_orders) != 0: # buy order exist
                pass
                # sell nothing


            result[product] = orders

        # traderData = state.toJSON()

        return result, conversions, traderData
