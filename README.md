# total-drama-island
We do not have dreams, we have goals

## setup for noobs:
1. [download vscode](https://code.visualstudio.com/)
2. [download python 3.12](https://www.python.org/downloads/)
3. connect vscode aan onze git repo
4. maak virtual environment (binnen vscode is makkelijkst) 
5. activeer virtual environment (en zorg dat je vscode ook de python uit de venv gebruikt niet je global install van python)
6. installeer requirements (```pip install -r requirements.txt```)

## requirements: 
- python 3.12
- requirments.txt (```pip install -r requirements.txt```)

## allowed python imports:
- pandas
- NumPy
- statistics
- math
- typing
- jsonpickle

## info:
**rounds:** 
The challenge will start on April 7th, 2025 at 9:00 AM CET. There are five rounds and each round lasts 72 hours.

Every round contains an algorithmic trading challenge. All rounds contain a manual trading challenge. Just like the algorithmic challenge, manual trading challenges last 72 hours to submit your (final) trade.
When the round ends, the last successfully processed submission will be locked in and processed for results.

0. Tutorial: February 24, 9:00 → April 7, 9:00
1. April 7, 9:00 → April 10, 9:00
2. April 10, 9:00 → April 13, 9:00
3. April 13, 9:00 → April 16, 9:00
4. April 16, 9:00 → April 19, 9:00
5. April 19, 9:00 → April 22, 9:00


**Tropical TV:**
Tropical TV is the archipelago’s daily news broadcast, hosted by a chatty cockatoo. The news broadcasts will keep you up to date with all developments around the archipelago. It contains all the information you need to navigate your way through the trading challenges that are thrown at you. It’s a great way to start your day!
Tropical TV episodes will be available at the beginning of every round. You will be notified as soon as a new Tropical TV episode becomes available.


**Simulation**
- alrgorithm will be written in ```run``` method of the ```Trader``` class
- every iteration of the simulation will execute the ```run``` method and be provided with  the ```TradingState``` object.
- ```TradingState``` contains: 
    - Contains overview of the trades of last iteration (of both alforithm and other participants)
    - Per product overview of outstanding buy/sell orders from bots
- **To Do:"finish info here**

## code snippets
**Trader class**
```python
# The Python code below is the minimum code that is required in a submission file:
# 1. The "datamodel" imports at the top. Using the typing library is optional.
# 2. A class called "Trader", this class name should not be changed.
# 3. A run function that takes a tradingstate as input and outputs a "result" dict.

from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
		"""
		Takes all buy and sell orders for all symbols as an input,
		and outputs a list of orders to be sent
		"""
        result = {}
        return result
```

**TradingState class**
The TradingState class holds all the important market information that an algorithm needs to make decisions about which orders to send. Below the definition is provided for the TradingState class:
```python
Time = int
Symbol = str
Product = str
Position = int

class TradingState(object):
   def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)
```

**Trade class**
Both the own_trades property and the market_trades property provide the traders with a list of trades per products. Every individual trade in each of these lists is an instance of the Trade class.
```python
Symbol = str
UserId = str

class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId = None, seller: UserId = None, timestamp: int = 0) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ")"
```

**OrderDepth class**
Provided by the TradingState class is also the OrderDepth per symbol. This object contains the collection of all outstanding buy and sell orders, or “quotes” that were sent by the trading bots, for a certain symbol. 
```python
class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}
```

**Observations class**
Observation details help to decide on eventual orders or conversion requests. There are two items delivered inside the TradingState instance:
1. Simple product to value dictionary inside plainValueObservations
2. Dictionary of complex ConversionObservation values for respective products. Used to place conversion requests from Trader class. Structure visible below.
```python
class ConversionObservation:

    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sugarPrice: float, sunlightIndex: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex
```

**Order class**
```python
Symbol = str

class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
```

## example trader class:
```python
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10;  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData

```