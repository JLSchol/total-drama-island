from datamodel import OrderDepth, TradingState, Order
from typing import List
import json


class Trader:
    def run(self, state: TradingState):
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
    
            if product == "SQUID_INK":
                if len(order_depth.sell_orders) != 0 and state.timestamp == 99600:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    if int(best_ask) and best_ask is not None:
                        orders.append(Order(product, best_ask, 1))
        
                # if len(order_depth.sell_orders) != 0 and state.timestamp == 99800:
                #     best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                #     if int(best_bid) and best_bid is not None:
                #         orders.append(Order(product, best_bid, -1))


            # if product == "RAINFOREST_RESIN":
            #     if len(order_depth.sell_orders) != 0 and state.timestamp == 99600:
            #         best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            #         if int(best_ask) and best_ask is not None:
            #             orders.append(Order(product, best_ask, 1))
        
            #     if len(order_depth.sell_orders) != 0 and state.timestamp == 99600:
            #         best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            #         if int(best_bid) and best_bid is not None:
            #             orders.append(Order(product, best_bid, -1))
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1 #ifyou want, you can convert an integer amount of your long or short position into moes, but need to cover the fees and trifs, send an integer or 0 if you don't want any conversion to happen
        return result, conversions, traderData
