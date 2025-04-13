# Round 2

## Algorithm challenge

In this second round, you’ll find that everybody on the archipelago loves to picnic. Therefore, in addition to the products from round one, two Picnic Baskets are now available as a tradable good. 

```python 
picic_content = {"PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                    "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2 }}
```
Aside from the Picnic Baskets, you can now also trade the three products individually on the island exchange. 
Position limits for the newly introduced products:

```python 
position_limits = {"RAINFOREST_RESIN": 50, "KELP": 50, 
                    "SQUID_INK": 50,
                    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
                    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100
                    }
```

## Manual challenge
Some shipping containers with valuables inside washed ashore. You get to choose a maximum of two containers to open and receive the valuable contents from. The first container you open is free of charge, but for the second one you will have to pay some SeaShells. Keep in mind that you are not the only one choosing containers and making a claim on its contents. You will have to split the spoils with all others that choose the same container. So, choose carefully. 

Here's a breakdown of how your profit from a container will be computed:
Every container has its **treasure multiplier** (up to 90) and number of **inhabitants** (up to 10) that will be choosing that particular container. The container’s total treasure is the product of the **base treasure** (10 000, same for all containers) and the container’s specific treasure multiplier. However, the resulting amount is then divided by the sum of the inhabitants that choose the same container and the percentage of opening this specific container of the total number of times a container has been opened (by all players). 

For example, if **5 inhabitants** choose a container, and **this container was chosen** **10% of the total number of times a container has been opened** (by all players), the prize you get from that container will be divided by 15. After the division, **costs for opening a container** apply (if there are any), and profit is what remains.