sma findings profit los:
- rainforest: 
    - sma_5: 814
    - sma_10: 858
    - **sma_20: 862**
    - *wsma_20*: 810
    - sma_40: 816
    - sma_80: 850
- kelp: 
    - sma_5: 212.8  
    - sma_10: 210.8
    - sma_20: **216.8**
    - *wsma_20*: 215.4
    - sma_40: 213.4
    - sma_80: 208.6
- squid_ink: 
    - **sma not profitable (for weighted and unweigted midprice values)
    - **higher values better since price is steadily decreasing**
    - sma_5: -5852.73
    - sma_10: -6544.47
    - sma_20: -4703.49
    - *wsma_20*: -4499.49
    - sma_40: -3365.13
    - sma_80: -2620.94


**simple moving average crossover tactic**
            product strategy_name  profit_factor  sharpe_ratio
0              KELP      sma10x30       0.006289    -11.150626
1  RAINFOREST_RESIN      sma10x30       0.033333    -11.988106
2         SQUID_INK      sma10x30       0.615385     -1.100875
3              KELP      sma15x30       0.038095     -8.149885
4  RAINFOREST_RESIN      sma15x30       0.240106     -5.738601
5         SQUID_INK      sma15x30       1.010471      0.020503
6              KELP      sma10x40       0.000000    -13.152894
7  RAINFOREST_RESIN      sma10x40       0.036913    -12.798961
8         SQUID_INK      sma10x40       0.907514     -0.179364
9              KELP      sma15x40       0.120690     -6.263926


**moving average crossover tactic with momentum**
        product          strategy_name  profit_factor  sharpe_ratio
1058  SQUID_INK  sma16x27xmom28x0.0045       3.172414      1.204433
536   SQUID_INK  sma15x27xmom28x0.0045       3.172414      1.204433
1580  SQUID_INK  sma17x27xmom28x0.0045       3.172414      1.204433
2102  SQUID_INK  sma18x27xmom28x0.0045       3.172414      1.204433
14    SQUID_INK  sma14x27xmom28x0.0045       3.172414      1.204433
2105  SQUID_INK   sma18x27xmom28x0.005       2.558824      1.006879
1061  SQUID_INK   sma16x27xmom28x0.005       2.558824      1.006879
1583  SQUID_INK   sma17x27xmom28x0.005       2.558824      1.006879
539   SQUID_INK   sma15x27xmom28x0.005       2.558824      1.006879
17    SQUID_INK   sma14x27xmom28x0.005       2.558824      1.006879


Could not make kelp work. hint later presented:
Squid inm can be a very volatile product, with large price swings, making a two sides market or carrying position can be risky.with large swings come large reversions. Squid ink shows more tendency to revert short term swings in price.

best props:
Recommended Simple Strategy: Mean-Reversion Band
Use a short-term SMA as a reference mean. Place orders just outside that mean — expecting reversion toward it.
Steps:

    Use short-term SMA (e.g., 5 or 10 ticks) of mid_price.

    Define a threshold deviation from SMA (e.g., ±2 ticks).

    If price drops below SMA - threshold → buy, expecting it to revert up.

    If price rises above SMA + threshold → sell, expecting it to revert down.

    Keep size small and close trades quickly — don’t carry.

**manual challange 2:**
Some shipping containers with valuables inside washed ashore. You get to choose a maximum of two containers to open and receive the valuable contents from. The first container you open is free of charge, but for the second one you will have to pay some SeaShells. Keep in mind that you are not the only one choosing containers and making a claim on its contents. You will have to split the spoils with all others that choose the same container. So, choose carefully. 

Here's a breakdown of how your profit from a container will be computed:
Every container has its **treasure multiplier** (up to 90) and number of **inhabitants** (up to 10) that will be choosing that particular container. The container’s total treasure is the product of the **base treasure** (10 000, same for all containers) and the container’s specific treasure multiplier. However, the resulting amount is then divided by the sum of the inhabitants that choose the same container and the percentage of opening this specific container of the total number of times a container has been opened (by all players). 

For example, if **5 inhabitants** choose a container, and **this container was chosen** **10% of the total number of times a container has been opened** (by all players), the prize you get from that container will be divided by 15. After the division, **costs for opening a container** apply (if there are any), and profit is what remains.

first container cost 0 seashels
second container costs 50.000 seashels

container nr, multiplier, inhabitants, nr other participants
1,10,1,?
2,80,6,?
3,37,3,?
4,90,10,?
5,31,2,?
6,17,1,?
7,50,4,?
8,20,2,?
9,73,4,?
10,89,8,?
