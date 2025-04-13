import numpy as np
import matplotlib.pyplot as plt

# Constants
sell_price = 320
low_range = (160, 200)
high_range = (250, 320)
total_turtles = 110
low_weight = 40 / total_turtles
high_weight = 70 / total_turtles

# Generate all possible bids (160 to 320)
second_bids = np.arange(160, 321)

# Function to calculate p factor
def penalty_factor(bid, avg_bid):
    if bid >= avg_bid:
        return 1
    return ((320 - avg_bid) / (320 - bid)) ** 3

# Probability functions
def prob_low_accept(bid):
    if low_range[0] <= bid <= low_range[1]:
        return (bid - low_range[0]) / (low_range[1] - low_range[0])
    elif bid < low_range[0] or bid >= high_range[0]:
        return 1
    return 0

def prob_high_accept(bid):
    if high_range[0] <= bid <= high_range[1]:
        return (bid - high_range[0]) / (high_range[1] - high_range[0])
    return 0

# Store results
avg_bids = np.arange(160, 321)
max_profits = []
best_bids = []

for avg_bid in avg_bids:
    profits = []

    for bid in second_bids:
        if 200 < bid < 250:
            continue  # Skip dead zone

        p_low = prob_low_accept(bid)
        p_high = prob_high_accept(bid)
        prob_accept = low_weight * p_low + high_weight * p_high
        p = penalty_factor(bid, avg_bid)
        profit_per_trade = sell_price - bid
        expected_profit = prob_accept * p * profit_per_trade
        profits.append((bid, expected_profit))

    if profits:
        best_bid, best_profit = max(profits, key=lambda x: x[1])
        max_profits.append(best_profit)
        best_bids.append(best_bid)
    else:
        max_profits.append(0)
        best_bids.append(None)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(avg_bids, max_profits, label="Max Expected Profit", linewidth=2)
plt.xlabel("Average Second Bid from Other Traders")
plt.ylabel("Max Expected Profit")
plt.title("Expected Profit vs. Average Second Bid")
plt.grid(True)
plt.legend()

# Optional: highlight stable region
stable_start = avg_bids[np.argmax(max_profits)]
plt.axvline(x=stable_start, color='orange', linestyle='--', label=f"Peak Profit Avg ≈ {stable_start}")
plt.legend()

plt.tight_layout()
plt.show()
