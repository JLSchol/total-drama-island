import itertools

# Define exchange rates (from -> to)
rates = {
    'snowballs': {'seashells': 0.72, 'snowballs': 1.0, 'pizzas': 1.45, 'silicon': 0.52},
    'pizzas': {'seashells': 0.48, 'snowballs': 0.7, 'pizzas': 1.0, 'silicon': 0.31},
    'silicon': {'seashells': 1.49, 'snowballs': 1.95, 'pizzas': 3.1, 'silicon': 1.0},
    'seashells': {'seashells': 1.0, 'snowballs': 1.34, 'pizzas': 1.98, 'silicon': 0.64}
}

currencies = ['snowballs', 'pizzas', 'silicon']  # Intermediate currencies

def simulate_trades(initial_amount=500000):
    bests = []
    for num_trades in [3, 4, 5, 6]:  # Try sequences of 3-5 trades
        # Generate all possible intermediate steps
        
        best = {'path': None, 'amount': 0}
        for intermediates in itertools.product(currencies, repeat=num_trades-2):
            path = ['seashells'] + list(intermediates) + ['seashells']
            amount = initial_amount
            
            # Apply each trade in sequence
            for i in range(len(path)-1):
                from_curr = path[i]
                to_curr = path[i+1]
                amount *= rates[from_curr][to_curr]
            
            # Update best found path
            if amount > best['amount']:
                best = {'path': path, 'amount': amount}
        bests.append(best)
    
    return bests

# Run simulation
results = simulate_trades()
for result in results:
    print(f"Best path: {' → '.join(result['path'])}")
    print(f"Final amount: {result['amount']:,.2f} SeaShells")
    print(f"Profit: {result['amount']/500000:.2f}x")
