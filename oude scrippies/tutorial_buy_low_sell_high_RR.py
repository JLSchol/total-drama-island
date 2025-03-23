from datamodel import OrderDepth, TradingState, Order
from typing import List


def buy_resin(order, product, position_limits, position, orders):
    lowest_ask_price, volume = min(order.items(), key=lambda x: x[0])
    buy_volume = volume*-1

    buy_space = position_limits[product] - position.get(product, 0)
    if buy_volume > buy_space:
        buy_volume = buy_space
    
    print(f"BUY {product}: for {lowest_ask_price}$ -- with size: {buy_volume}")
    orders.append(Order(product, lowest_ask_price, buy_volume))
    return orders

def sell_resin(order, product, position_limits, position, orders):
    highest_ask_price, volume = max(order.items(), key=lambda x: x[0])
    sell_volume = -1

    sell_space = -1*position_limits[product] - position.get(product, 0)
    if abs(sell_volume) > abs(sell_space):
        sell_volume = sell_space
    
    print(f"sell {product}: for {highest_ask_price}$ -- with size: {sell_volume}")
    orders.append(Order(product, highest_ask_price, sell_volume))
    return orders

class Trader:
    def run(self, state: TradingState):
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
                if product == "RAINFOREST_RESIN":
                    orders = buy_resin(order_depth.sell_orders, product, position_limits, state.position, orders)

            if len(order_depth.buy_orders) != 0: # buy order exist
                if product == "RAINFOREST_RESIN":
                    orders = sell_resin(order_depth.sell_orders, product, position_limits, state.position, orders)
                    pass
            result[product] = orders

        return result, conversions, traderData
