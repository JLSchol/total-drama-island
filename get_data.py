from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order





class Trader:

    def get_listings(self):
        pass

    def get_order(self):
        pass

    def get_order_depth(self):
        pass

    def get_trade(self):
        pass

    def get_trading_state(self):
        pass

    def get_prosperity_encoder(selft):
        pass

    def get_data():
        pass

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        #  index	timestamp	commodity	buy/sell	price	volume	position
        
        for commodity, value in state.order_depths.items():
            for buy_sell, price_vol_pair in value.items():
                for price, volume in price_vol_pair.items():
                    txt_line = "{},{},{},{},{},{}".format(   state.timestamp,
                                                                commodity,
                                                                buy_sell,
                                                                price,
                                                                volume,
                                                                state.position  )
                    print(txt_line)





        # txt_line = "{},{},{}".format(state.timestamp,                              
        #                           state.listings["PEARLS"].symbol,                              
        #                           state.listings["PEARLS"].denomination
        #                         )
        # print(txt_line)
































        
                
        return result
