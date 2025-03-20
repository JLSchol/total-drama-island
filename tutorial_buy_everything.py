from datamodel import OrderDepth, TradingState, Order
from typing import List


def buy_all(order_depth, product, position_limits, position, orders):

    highest_ask_price, _ = max(order_depth.sell_orders.items(), key=lambda x: x[0])

    total_amount_asks = 0
    for vol in order_depth.sell_orders.values():
        total_amount_asks += vol

    buy_amount = total_amount_asks *-1

    buy_room = position_limits[product] - position.get(product, 0)
    if buy_amount > buy_room:
        buy_amount = buy_room

    print(f"BUY {product}: for {highest_ask_price}$ -- with size: {buy_amount}")
    orders.append(Order(product, highest_ask_price, buy_amount))

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
                orders = buy_all(order_depth, product, position_limits, state.position, orders)

            if len(order_depth.buy_orders) != 0: # buy order exist
                pass

            result[product] = orders

        return result, conversions, traderData
