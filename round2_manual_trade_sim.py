import pandas as pd

# Container data (static values)
container_data = [
    {"Container": 1, "Multiplier": 10, "Inhabitants": 1},
    {"Container": 2, "Multiplier": 80, "Inhabitants": 6},
    {"Container": 3, "Multiplier": 37, "Inhabitants": 3},
    {"Container": 4, "Multiplier": 90, "Inhabitants": 10},
    {"Container": 5, "Multiplier": 31, "Inhabitants": 2},
    {"Container": 6, "Multiplier": 17, "Inhabitants": 1},
    {"Container": 7, "Multiplier": 50, "Inhabitants": 4},
    {"Container": 8, "Multiplier": 20, "Inhabitants": 2},
    {"Container": 9, "Multiplier": 73, "Inhabitants": 4},
    {"Container": 10, "Multiplier": 89, "Inhabitants": 8}
]

# Create DataFrame
df = pd.DataFrame(container_data)

# Default % of traders (Nash Equilibrium values from earlier)
default_pct_traders = {
    1: 1.5,
    2: 9.3,
    3: 6.5,
    4: 9.0,
    5: 7.2,
    6: 3.8,
    7: 16.4,
    8: 3.1,
    9: 32.0,
    10: 10.2
}

def calculate_shares(df, pct_traders=None, total_traders=3000):
    """
    Calculate 'My Share' based on % of traders picking each container.
    
    Args:
        df: DataFrame with container data
        pct_traders: Dictionary of {container: %} (defaults to Nash values)
        total_traders: Total number of traders (default 3000)
    
    Returns:
        DataFrame with added '% of Traders' and 'My Share' columns
    """
    if pct_traders is None:
        pct_traders = default_pct_traders
    
    # Add % of traders column
    df['% of Traders'] = df['Container'].map(pct_traders)
    
    # Calculate My Share
    base_treasure = 10000
    df['My Share'] = (base_treasure * df['Multiplier']) / (df['Inhabitants'] + df['% of Traders'])
    
    return df

# Example usage
result_df = calculate_shares(df)
print(result_df[['Container', 'Multiplier', 'Inhabitants', '% of Traders', 'My Share']].round(2))

# Example of modifying trader distribution
custom_pct = {
    1: 1.0,
    2: 10.0,
    3: 9.0,
    4: 5.0,
    5: 9.0,
    6: 5.0,
    7: 18.0,
    8: 5.0,
    9: 30.0,
    10: 6.0
}
print("\nWith custom trader distribution:")
custom_df = calculate_shares(df, custom_pct)
print(custom_df[['Container', 'Multiplier', 'Inhabitants', '% of Traders', 'My Share']].round(2))
print(sum(custom_pct.values()))