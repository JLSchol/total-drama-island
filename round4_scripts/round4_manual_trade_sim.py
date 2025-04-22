import pandas as pd
# seal or no seal
# suitcase data (static values)
suitcase_data = [
    {"suitcase": 1, "Multiplier": 80, "contestants": 6},
    {"suitcase": 2, "Multiplier": 50, "contestants": 4},
    {"suitcase": 3, "Multiplier": 83, "contestants": 7},
    {"suitcase": 4, "Multiplier": 31, "contestants": 2},
    {"suitcase": 5, "Multiplier": 60, "contestants": 4},

    {"suitcase": 6, "Multiplier": 89, "contestants": 8},
    {"suitcase": 7, "Multiplier": 10, "contestants": 1},
    {"suitcase": 8, "Multiplier": 37, "contestants": 3},
    {"suitcase": 9, "Multiplier": 70, "contestants": 4},
    {"suitcase": 10, "Multiplier": 90, "contestants": 10},

    {"suitcase": 1, "Multiplier": 17, "contestants": 1},
    {"suitcase": 12, "Multiplier": 40, "contestants": 3},
    {"suitcase": 13, "Multiplier": 73, "contestants": 4},
    {"suitcase": 14, "Multiplier": 100, "contestants": 15},
    {"suitcase": 15, "Multiplier": 20, "contestants": 2},

    {"suitcase": 16, "Multiplier": 41, "contestants": 3},
    {"suitcase": 17, "Multiplier": 79, "contestants": 5},
    {"suitcase": 18, "Multiplier": 23, "contestants": 2},
    {"suitcase": 19, "Multiplier": 47, "contestants": 3},
    {"suitcase": 20, "Multiplier": 30, "contestants": 2}
]

# Create DataFrame
df = pd.DataFrame(suitcase_data)
df2 = df.copy()

# Default % of traders (Nash Equilibrium values from earlier)
default_pct_traders = {
    1: 5.0,
    2: 5.0,
    3: 5.0,
    4: 5.0,
    5: 5.0,
    6: 5.0,
    7: 5.0,
    8: 5.0,
    9: 5.0,
    10: 5.0,
    11: 5.0,
    12: 5.0,
    13: 5.0,
    14: 5.0,
    15: 5.0,
    16: 5.0,
    17: 5.0,
    18: 5.0,
    19: 5.0,
    20: 5.0
}

def calculate_shares(df, pct_traders=None, total_traders=3000):
    """
    Calculate 'My Share' based on % of traders picking each suitcase.
    
    Args:
        df: DataFrame with suitcase data
        pct_traders: Dictionary of {suitcase: %} (defaults to Nash values)
        total_traders: Total number of traders (default 3000)
    
    Returns:
        DataFrame with added '% of Traders' and 'My Share' columns
    """
    #

    if pct_traders is None:
        pct_traders = default_pct_traders
    
    df['effectivive_M'] = df['Multiplier'] / df['contestants']

    # Add % of traders column
    df['% of Traders'] = df['suitcase'].map(pct_traders)
    
    # Calculate My Share
    base_treasure = 10000
    df['My Share'] = (base_treasure * df['Multiplier']) / (df['contestants'] + df['% of Traders'])

    
    return df

# Example usage
result_df = calculate_shares(df)
print(result_df[['suitcase', 'Multiplier', 'contestants', '% of Traders', 'My Share']].round(2))

# Example of modifying trader distribution
custom_pct = {
    1: 5.0,
    2: 5.0,
    3: 5.0,
    4: 5.0,
    5: 5.0,
    6: 5.0,
    7: 5.0,
    8: 5.0,
    9: 5.0,
    10: 5.0,
    11: 5.0,
    12: 5.0,
    13: 5.0,
    14: 5.0,
    15: 5.0,
    16: 5.0,
    17: 5.0,
    18: 5.0,
    19: 5.0,
    20: 5.0
}


print("\nWith custom trader distribution:")
custom_df = calculate_shares(df2, custom_pct)
print(custom_df[['suitcase', 'Multiplier', 'contestants', 'effectivive_M', '% of Traders', 'My Share']].round(2))
print(sum(custom_pct.values()))