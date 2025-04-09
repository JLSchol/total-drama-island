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