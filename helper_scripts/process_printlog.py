import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_df(round_name, log_name):
    analyze_dir = log_name

    this_dir = os.path.dirname(__file__)
    printlogs_file_path = os.path.join(this_dir, "..", "logs",round_name, analyze_dir,"processed", "printlogs.csv") 

    df = pd.read_csv(printlogs_file_path)
    return df

def save_df(df, round_name, analyze_dir, csv_name):
    this_dir = os.path.dirname(__file__)
    save_dir = os.path.join(this_dir, "..", "logs", round_name, analyze_dir, "analyzed")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df.to_csv(os.path.join(save_dir, csv_name), index=False)

def add_simpel_moving_average(df, column_name, window):
    ma_column_name = f"{column_name}_sma_{window}"
    
    # Group by product and apply rolling mean
    df[ma_column_name] = df.groupby('product')[column_name] \
                            .transform(lambda x: x.rolling(window=window).mean())
    
    return df, ma_column_name

def add_sma_crossover_signal(df, sma_fast_col, sma_slow_col, id_base_name):
    """
    Adds a signal column based on SMA crossover strategy.

    Parameters:
        df (pd.DataFrame): DataFrame containing the fast and slow SMA columns.
        sma_fast_col (str): Column name of the fast SMA.
        sma_slow_col (str): Column name of the slow SMA.

    Returns:
        pd.DataFrame: DataFrame with added 'sma_signal' column.
    """

    signal_name = f"{id_base_name}_signal"
    df[signal_name] = 0  # Default hold

    df.loc[df[sma_fast_col] > df[sma_slow_col], signal_name] = 1   
    df.loc[df[sma_fast_col] < df[sma_slow_col], signal_name] = -1  

    return df, signal_name

def add_momentum_crossover_filter_signal(df, crossover_signal_name, id_base_name, momentum_period=30, momentum_threshold=0.0):
    """
    Applies a momentum filter to an existing SMA crossover signal.

    Parameters:
        df (pd.DataFrame): DataFrame with price and SMA crossover signal.
        crossover_signal_name (str): The name of the existing crossover signal column.
        momentum_period (int): The lookback period for momentum calculation.
        momentum_threshold (float): Minimum absolute momentum required to confirm signal.
        id_base_name (str): Base name for the new filtered signal column.

    Returns:
        df (pd.DataFrame): Updated DataFrame with filtered signal column.
        signal_name (str): Name of the new signal column.
    """
    signal_name = f"{id_base_name}_signal"
    df[signal_name] = np.nan  # Start with NaNs

    # Calculate momentum per product (e.g., 30-bar price rate of change)
    df['momentum'] = df.groupby('product')['mid_price'].transform(lambda x: x.pct_change(periods=momentum_period))

    # Apply the momentum filter
    def filter_logic(row):
        sig = row[crossover_signal_name]
        mom = row['momentum']

        if sig == 1 and mom > momentum_threshold:
            return 1
        elif sig == -1 and mom < -momentum_threshold:
            return -1
        else:
            return np.nan  # Reject signal, hold current

    df[signal_name] = df.apply(filter_logic, axis=1)

    # Fill forward to maintain a position always
    df[signal_name] = df.groupby('product')[signal_name].ffill().bfill()

    return df, signal_name

def simulate_trades_from_signals(df, signal_col, strategy_name, price_col='mid_price', df_simulation=None):
    """
    Simulates long and short trades based on the signal column, 
    calculates PnL for each trade, and adds relevant data to the provided simulation DataFrame.
    
    This version handles multiple products by grouping the DataFrame by 'product'.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the signal and price data.
        price_col (str): The column with price data (typically 'mid_price').
        signal_col (str): The column with trade signals (1 for buy, -1 for sell).
        strategy_name (str): Name of the strategy being applied (e.g., 'sma_crossover').
        df_simulation (pd.DataFrame): The DataFrame holding simulation data for different strategies.
    
    Returns:
        pd.DataFrame: DataFrame with additional columns for entry/exit price, PnL, and position for each strategy.
    """
    df = df.copy()

    # Initialize columns specific to the current strategy
    df['position'] = 0  # 1 for long, -1 for short, 0 for no position
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['pnl'] = np.nan
    df['signal_shifted'] = df.groupby('product')[signal_col].shift(1)
    df['strategy_name'] = strategy_name

    # Iterate through each product group
    for product, group in df.groupby('product'):
        position = None
        entry_price = None
        
        last_index = None  # track last index in case we need to force close
        for i, row in group.iterrows():
            prev_signal = row['signal_shifted']
            curr_signal = row[signal_col]
            price = row[price_col]
            last_index = i  # keep track of last row index

            # Exit current position if any
            if position == 'long' and curr_signal == -1:
                pnl = price - entry_price
                df.at[i, 'exit_price'] = price
                df.at[i, 'pnl'] = pnl
                df.at[i, 'position'] = 0
                position = None
                entry_price = None

            elif position == 'short' and curr_signal == 1:
                pnl = entry_price - price
                df.at[i, 'exit_price'] = price
                df.at[i, 'pnl'] = pnl
                df.at[i, 'position'] = 0
                position = None
                entry_price = None

            # Enter new position if any
            if curr_signal == 1 and position is None:
                position = 'long'
                entry_price = price
                df.at[i, 'position'] = 1
                df.at[i, 'entry_price'] = price

            elif curr_signal == -1 and position is None:
                position = 'short'
                entry_price = price
                df.at[i, 'position'] = -1
                df.at[i, 'entry_price'] = price

        # --- Final forced exit if still in position ---
        if position is not None and last_index is not None:
            final_price = group.loc[last_index, price_col]
            if position == 'long':
                pnl = final_price - entry_price
                df.at[last_index, 'exit_price'] = final_price
                df.at[last_index, 'pnl'] = pnl
                df.at[last_index, 'position'] = 0

            elif position == 'short':
                pnl = entry_price - final_price
                df.at[last_index, 'exit_price'] = final_price
                df.at[last_index, 'pnl'] = pnl
                df.at[last_index, 'position'] = 0

    # Create a DataFrame to hold the results for this strategy
    strategy_results = df[['product', 'timestamp', 'strategy_name', price_col, signal_col,
                           'signal_shifted', 'position', 'entry_price', 'exit_price', 'pnl']]

    # If df_simulation is None, initialize it
    if df_simulation is None:
        df_simulation = strategy_results
    else:
        # Append new strategy results to the simulation DataFrame
        df_simulation = pd.concat([df_simulation, strategy_results], axis=0, ignore_index=True)
    # Calculate cumulative PnL per product and strategy

    df_simulation = df_simulation.sort_values(by=['timestamp', 'product', 'strategy_name']).reset_index(drop=True)

    # Fill NaNs in pnl with 0 for clean cumsum (but keep entry/exit NaNs unchanged)
    # Calculate cumulative PnL per product and strategy, starting at 0
    df_simulation['pnl_clean'] = df_simulation['pnl'].fillna(0)
    df_simulation['cum_pnl'] = (
        df_simulation
        .groupby(['product', 'strategy_name'])['pnl_clean']
        .cumsum()
    )
    df_simulation = df_simulation.drop(columns='pnl_clean')
    

    return df_simulation

def add_profit_factor(df_metrics, df_simulation, pnl_col='pnl', product_col='product', strategy_col='strategy_name'):
    """
    Adds profit factor per product-strategy pair to df_metrics.
    """

    if df_metrics is None or df_metrics.empty:
        df_metrics = pd.DataFrame(columns=[product_col, strategy_col, 'profit_factor'])

    for (product, strategy), group in df_simulation.groupby([product_col, strategy_col]):
        wins = group[group[pnl_col] > 0][pnl_col].sum()
        losses = group[group[pnl_col] < 0][pnl_col].sum()

        total_wins = wins if wins > 0 else 0
        total_losses = abs(losses) if losses < 0 else 0

        if total_losses > 0:
            profit_factor = total_wins / total_losses
        elif total_wins > 0 and total_losses == 0:
            profit_factor = np.inf
        else:
            profit_factor = np.nan

        # Check if this product-strategy already exists in df_metrics
        mask = (df_metrics[product_col] == product) & (df_metrics[strategy_col] == strategy)

        if df_metrics[mask].empty: # if the product-strategy pair does not exist
            # Create a new row with the profit factor and concat to existing metric df
            df_metrics = pd.concat([
                df_metrics,
                pd.DataFrame([{product_col: product, strategy_col: strategy, 'profit_factor': profit_factor}])
            ], ignore_index=True)
        else: # if pair exists, just update the value
            df_metrics.loc[mask, 'profit_factor'] = profit_factor

    return df_metrics

def add_sharpe_ratio(df_metrics, df_simulation, pnl_col='pnl', product_col='product', strategy_col='strategy_name'):
    """
    Calculate the Sharpe Ratio for each (product, strategy) and update df_metrics accordingly.

    Parameters:
        df_metrics (pd.DataFrame): DataFrame to store metrics per product-strategy.
        df_simulation (pd.DataFrame): Simulation data containing PnL and strategy info.
        pnl_col (str): The column containing profit/loss values.
        product_col (str): The column that contains product names.
        strategy_col (str): The column that contains strategy names.

    Returns:
        pd.DataFrame: Updated df_metrics with Sharpe Ratios.
    """
    # If df_metrics is None, initialize it
    if df_metrics is None:
        df_metrics = pd.DataFrame(columns=[product_col, strategy_col, 'sharpe_ratio'])

    # Group by product and strategy
    for (product, strategy), group in df_simulation.groupby([product_col, strategy_col]):
        # Filter valid pnl values
        pnl = group[pnl_col].dropna()
        
        if not pnl.empty:
            mean_pnl = pnl.mean()
            std_pnl = pnl.std()
            steps = len(pnl)

            # Avoid division by zero
            if std_pnl > 0:
                sharpe_ratio = mean_pnl / std_pnl * np.sqrt(steps)
            else:
                sharpe_ratio = np.nan
        else:
            sharpe_ratio = np.nan

        # Check if this product-strategy already exists in df_metrics
        mask = (df_metrics[product_col] == product) & (df_metrics[strategy_col] == strategy)

        if df_metrics[mask].empty:
            # Append new row
            df_metrics = pd.concat([
                df_metrics,
                pd.DataFrame([{product_col: product, strategy_col: strategy, 'sharpe_ratio': sharpe_ratio}])
            ], ignore_index=True)
        else:
            # Update existing row
            df_metrics.loc[mask, 'sharpe_ratio'] = sharpe_ratio

    return df_metrics

def analyze_signal(df, signal_name, strat_name, price_col_name, df_simulation=None, df_metrics=None):
    """
    Analyzes the crossover of two SMAs and simulates trades based on the crossover signals.

    Parameters:
        df (pd.DataFrame): DataFrame containing the price data and SMAs.
        sma_fast (str): Column name for the fast SMA.
        sma_slow (str): Column name for the slow SMA.
        strat_name (str): Name of the strategy being applied.
        price_col_name (str): Column name for the price data.
        df_simulation (pd.DataFrame, optional): DataFrame to hold simulation results. Defaults to None.
        df_metrics (pd.DataFrame, optional): DataFrame to hold metrics. Defaults to None.

    Returns:
        pd.DataFrame: Updated df_simulation with trade data.
        pd.DataFrame: Updated df_metrics with performance metrics.
    """

    df_simulation = simulate_trades_from_signals(df, signal_name, strat_name, price_col_name, df_simulation)
    df_metrics = add_profit_factor(df_metrics, df_simulation)
    df_metrics = add_sharpe_ratio(df_metrics, df_simulation)
    return df, df_simulation, df_metrics

def plot_crossover(df, df_simulation, df_metrics, plot_info, save_dir=None):
    """
    Plots mid_price, crossover points, entry/exit signals, and strategy metrics.

    Parameters:
        df (pd.DataFrame): DataFrame containing the price data and SMA values.
        df_simulation (pd.DataFrame): DataFrame containing the simulation results (entry/exit signals, PnL, etc.).
        df_metrics (pd.DataFrame): DataFrame containing the performance metrics (Profit Factor, Sharpe Ratio).
        plot_info (dict): Information on which products and strategies to plot.
    """
    products = plot_info["products"]
    strategies = plot_info["strategies"]

    for product in products:
        # Create a figure for each product
        fig, axes = plt.subplots(len(strategies), 1, figsize=(14, 4 * len(strategies)), sharex=True)
        if len(strategies) == 1:
            axes = [axes]  # Ensure axes is always iterable

        for idx, strategy in enumerate(strategies):
            ax = axes[idx]
            strategy_name = strategy["name"]
            price_col = strategy["price"]
            fast_sma = strategy["fast"]
            slow_sma = strategy["slow"]

            # Filter data for this product
            df_plot = df[df['product'] == product]
            df_strat = df_simulation[(df_simulation['product'] == product) & (df_simulation['strategy_name'] == strategy_name)]

            # Plot mid price
            ax.plot(df_plot['timestamp'], df_plot[price_col], label='Mid Price', color='black', alpha=0.5)

            # Plot the two SMAs
            ax.plot(df_plot['timestamp'], df_plot[fast_sma], label=f'{fast_sma} (Fast)', alpha=0.6)
            ax.plot(df_plot['timestamp'], df_plot[slow_sma], label=f'{slow_sma} (Slow)', alpha=0.6)

            # Mark entries and exits (with vertical lines)
            entries = df_strat[df_strat['entry_price'].notna()]
            exits = df_strat[df_strat['exit_price'].notna()]

            # Avoid repeating legend entries for Entry/Exit lines
            entry_added, exit_added = False, False
            for _, entry in entries.iterrows():
                ax.axvline(entry['timestamp'], color='green', linestyle='-', alpha=1, label='Entry' if not entry_added else "")
                entry_added = True

            for _, exit in exits.iterrows():
                ax.axvline(exit['timestamp'], color='red', linestyle=':', alpha=1, label='Exit' if not exit_added else "")
                exit_added = True

            # Shade background between entry and exit
            in_position = False
            entry_time, position = None, None

            for _, row in df_strat.iterrows():
                # 🟥 Exit logic comes first
                if in_position and pd.notna(row['exit_price']):
                    exit_time = row['timestamp']
                    pnl = row['pnl']
                    color = 'lightgreen' if position == 1 else 'lightcoral'
                    ax.axvspan(entry_time, exit_time, color=color, alpha=0.2)

                    # Midpoint label
                    midpoint_time = entry_time + (exit_time - entry_time) / 2
                    price_mid_idx = df_plot['timestamp'].searchsorted(midpoint_time)
                    if 0 <= price_mid_idx < len(df_plot):
                        price_mid = df_plot.iloc[price_mid_idx][price_col]
                        ax.text(midpoint_time, price_mid, f'PnL: {pnl:.2f}',
                                ha='center', va='center', fontsize=9,
                                color='black', bbox=dict(facecolor='white', alpha=0.7))

                    in_position = False
                    entry_time, position = None, None  # reset

                # 🟩 Entry logic comes second
                if pd.notna(row['entry_price']):
                    in_position = True
                    entry_time = row['timestamp']
                    position = row['position']

            # Fetch performance metrics from df_metrics
            metric_row = df_metrics[(df_metrics['product'] == product) & (df_metrics['strategy_name'] == strategy_name)]
            profit_factor = metric_row['profit_factor'].values[0] if not metric_row.empty else None
            sharpe_ratio = metric_row['sharpe_ratio'].values[0] if not metric_row.empty else None
            total_pnl = df_strat['cum_pnl'].iloc[-1]

            # Title and text
            ax.set_title(f'{product} - {strategy_name}')
            textstr = f'Profit Factor: {profit_factor:.4f}\nSharpe Ratio: {sharpe_ratio:.4f}\nTotal PnL: {total_pnl:.2f}'
            ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            ax.legend()

            ax.grid(True)

        plt.tight_layout()
        # plt.subplots_adjust(top=0.9)  # Adjust top margin to make room for suptitle
        plt.suptitle(f'Product: {product}', fontsize=16, y=1.02)

        # Save the plot if required
        if save_dir:
            # Save the figure
            filename = f'{product}_{"_".join([s["name"] for s in strategies])}.png'
            file_path = os.path.join(save_dir, filename)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
            plt.close(fig)

def plot_df_columns(df, columns, products, kind='line', titles=None, figsize=(12, 4)):
    """
    Plots specified columns from a DataFrame with a subplot for each product.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list of str): Column names to plot.
        products (list of str): List of products to filter and plot.
        kind (str or list): Plot type(s) per product ('line', 'bar', etc.).
        titles (str or list): Plot titles per product. If None, uses default format.
        figsize (tuple): Size of each subplot.

    Returns:
        None (displays the plots)
    """
    if isinstance(products, str):
        products = [products]

    n = len(products)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(figsize[0], figsize[1] * n), sharex=True)

    if n == 1:
        axes = [axes]  # ensure iterable for single subplot

    # Normalize kind and titles into lists if needed
    if isinstance(kind, str):
        kind = [kind] * n
    if titles is None:
        titles = [f"{product}" for product in products]
    elif isinstance(titles, str):
        titles = [titles] * n

    for i, product in enumerate(products):
        product_df = df[df['product'] == product]

        if product_df.empty:
            print(f"Warning: No data for product '{product}'")
            continue

        product_df[columns].plot(ax=axes[i], kind=kind[i], title=titles[i])
        axes[i].set_ylabel("Value")
        axes[i].grid(True)

    plt.xlabel("Index")
    plt.tight_layout()

def single_crossover_analyse(df, product, fast_l, slow_l, mom_period, mom_threshold, price_col='mid_price'):
    # calculated columns needed for signal (sma's)
    df, sma_name_fast = add_simpel_moving_average(df, price_col, fast_l)
    df, sma_name_slow = add_simpel_moving_average(df, price_col, slow_l)

    # define name strategy and params
    cross_strat_name = f"sma{fast_l}x{slow_l}"
    cross_strat_params = {"name":cross_strat_name, "price": price_col, "fast": sma_name_fast, "slow": sma_name_slow}

    # create the actual signal
    df, crossover_signal_name = add_sma_crossover_signal(df, sma_name_fast, sma_name_slow, cross_strat_params["name"])
    # add the momentum filter -df, crossover_signal_name, id_base_name, momentum_period=30, momentum_threshold=0.0-
    cross_mom_strat_name = f"{cross_strat_name}xmom"
    df, signal_momentum_name = add_momentum_crossover_filter_signal(df, crossover_signal_name, 
                                                                    cross_mom_strat_name, mom_period, mom_threshold)

    # analyse crossover
    df, df_simulation, df_metrics = analyze_signal(df, 
                                        signal_momentum_name, cross_strat_params["name"], cross_strat_params["price"], 
                                        None, None)

    # plot
    # cross_strat_params["name"] = cross_mom_strat_name #update name for plot title
    plot_info = { 
                "products": [product],
                "strategies": [cross_strat_params]}
    plot_crossover(df, df_simulation, df_metrics, plot_info)

    # print
    print(df.loc[df['product'] == 'SQUID_INK'].head(10))
    print(df_simulation.loc[df_simulation['product'] == 'SQUID_INK'].head(10))
    print(df_metrics.loc[df_metrics['product'] == 'SQUID_INK'].head(10))

def all_crossover_in_sample_analysis(df):
    # Load the DataFrame and append some sma's
    df, sma_name_5 = add_simpel_moving_average(df, "mid_price", 5)
    df, sma_name_10 = add_simpel_moving_average(df, "mid_price", 10)
    df, sma_name_15 = add_simpel_moving_average(df, "mid_price", 15)
    df, sma_name_20 = add_simpel_moving_average(df, "mid_price", 20)
    df, sma_name_30 = add_simpel_moving_average(df, "mid_price", 30)
    df, sma_name_40 = add_simpel_moving_average(df, "mid_price", 40)

    # define what to plot
    sma5x20_strat = {"name":"sma5x20", "price": "mid_price", "fast": sma_name_5, "slow": sma_name_20}
    sma10x30_strat = {"name":"sma10x30", "price": "mid_price", "fast": sma_name_10, "slow": sma_name_30}
    sma15x30_strat = {"name":"sma15x30", "price": "mid_price", "fast": sma_name_15, "slow": sma_name_30}
    sma10x40_strat = {"name":"sma10x40", "price": "mid_price", "fast": sma_name_10, "slow": sma_name_40}
    sma15x40_strat = {"name":"sma15x40", "price": "mid_price", "fast": sma_name_15, "slow": sma_name_40}
    sma20x40_strat = {"name":"sma20x40", "price": "mid_price", "fast": sma_name_20, "slow": sma_name_40}

    # simulate trades and calculate metrics
    df, df_signal = add_sma_crossover_signal(df, sma5x20_strat["fast"], sma5x20_strat["slow"], sma5x20_strat["name"])
    df, df_simulation, df_metrics = analyze_signal(df, 
                                        df_signal, sma5x20_strat["name"], sma5x20_strat["price"], 
                                        None, None)
    

    df, df_signal = add_sma_crossover_signal(df, sma10x30_strat["fast"], sma10x30_strat["slow"], sma10x30_strat["name"])
    df, df_simulation, df_metrics = analyze_signal(df, 
                                        df_signal, sma10x30_strat["name"], sma10x30_strat["price"], 
                                        df_simulation, df_metrics)
    

    df, df_signal = add_sma_crossover_signal(df, sma15x30_strat["fast"], sma15x30_strat["slow"], sma15x30_strat["name"])
    df, df_simulation, df_metrics = analyze_signal(df, 
                                        df_signal, sma15x30_strat["name"], sma15x30_strat["price"], 
                                        df_simulation, df_metrics)
    

    df, df_signal = add_sma_crossover_signal(df, sma10x40_strat["fast"], sma10x40_strat["slow"], sma10x40_strat["name"])
    df, df_simulation, df_metrics = analyze_signal(df, 
                                        df_signal, sma10x40_strat["name"], sma10x40_strat["price"], 
                                        df_simulation, df_metrics)
    

    df, df_signal = add_sma_crossover_signal(df, sma15x40_strat["fast"], sma15x40_strat["slow"], sma15x40_strat["name"])
    df, df_simulation, df_metrics = analyze_signal(df, 
                                        df_signal, sma15x40_strat["name"], sma15x40_strat["price"], 
                                        df_simulation, df_metrics)
    

    df, df_signal = add_sma_crossover_signal(df, sma20x40_strat["fast"], sma20x40_strat["slow"], sma20x40_strat["name"])
    df, df_simulation, df_metrics = analyze_signal(df, 
                                        df_signal, sma20x40_strat["name"], sma20x40_strat["price"], 
                                        df_simulation, df_metrics)

    
    # plot/print trades, metrics, and strategies
    products = ['RAINFOREST_RESIN', 'KELP', 'SQUID_INK']
    round = "round1"
    directory = "2504071725_sma20_sma20_sma20"
    save_dir = os.path.join(os.path.dirname(__file__), "..", "logs", round, directory, "analyzed")

    # split in two plots otherwise to big
    plot_info = { 
                "products": products,
                "strategies": [sma5x20_strat, sma10x30_strat, sma15x30_strat]}
    plot_crossover(df, df_simulation, df_metrics, plot_info, save_dir)

    plot_info = { 
                "products": products,
                "strategies": [sma10x40_strat, sma15x40_strat, sma20x40_strat]}
    
    plot_crossover(df, df_simulation, df_metrics, plot_info, save_dir)

    print(df.head(5))
    print()
    print(df_simulation.head(5))
    print()
    print(df_metrics.head(6*3))

    # save to csv ()
    save_df(df, round, directory,"df_cross.csv")
    save_df(df_simulation, round, directory,"df_cross_sim.csv")
    save_df(df_metrics, round, directory,"df_cross_metric.csv")
    # call show last

def analyse_cross_momentum_params(df, params, price_col='mid_price'):

    df_original = df.copy()
    df_simulation = None
    df_metrics = None
    for f,s,m,t in params:
       
        # calculated columns needed for signal (sma's)
        df, sma_name_fast = add_simpel_moving_average(df, price_col, f)
        df, sma_name_slow = add_simpel_moving_average(df, price_col, s)

        # define name strategy and params
        cross_strat_name = f"sma{f}x{s}"
        cross_strat_params = {"name":cross_strat_name, "price": price_col, "fast": sma_name_fast, "slow": sma_name_slow}

        # create the actual signal
        df, crossover_signal_name = add_sma_crossover_signal(df, sma_name_fast, sma_name_slow, cross_strat_params["name"])
        # add the momentum filter -df, crossover_signal_name, id_base_name, momentum_period=30, momentum_threshold=0.0-
        cross_mom_strat_name = f"{cross_strat_name}xmom{m}x{t}"
        df, signal_momentum_name = add_momentum_crossover_filter_signal(df, crossover_signal_name, 
                                                                        cross_mom_strat_name, m, t)

        # analyse crossover
        df, df_simulation, df_metrics = analyze_signal(df, 
                                            signal_momentum_name, cross_mom_strat_name, cross_strat_params["price"], 
                                            df_simulation, df_metrics)
        # reset
        df = df_original.copy()
        df_simulation = None

    return df_metrics

def analyse_and_inspect_cross_momentum():
    grid = generate_param_grid(16, 31, 32, 0.0035, 2, 4, 4, 0.001)
    df_metrics = analyse_cross_momentum_params(df, grid)
    df_squid = df_metrics.loc[df_metrics['product'] == 'SQUID_INK']
    df_squid = df_squid.sort_values(by=['profit_factor'], ascending=False)
    df_squid.to_csv("squid_metrics.csv", index=False)
    print(df_squid.head(100))

def generate_param_grid(
    fast_l_base=16, slow_l_base=31, 
    mom_period_base=31, mom_threshold_base=0.0035,
    delta_fast=2, delta_slow=4, 
    delta_mom_period=4, delta_mom_thresh=0.001):

    fast_ls = range(fast_l_base - delta_fast, fast_l_base + delta_fast + 1, 1)
    slow_ls = range(slow_l_base - delta_slow, slow_l_base + delta_slow + 1, 2)
    mom_periods = range(mom_period_base - delta_mom_period, mom_period_base + delta_mom_period + 1, 1)
    mom_thresholds = np.arange(
        mom_threshold_base - delta_mom_thresh, 
        mom_threshold_base + delta_mom_thresh + 0.0005, 
        0.0005
    )

    print(f"fast: {list(fast_ls)}, slow: {list(slow_ls)}, mom_periods: {list(mom_periods)}, mom_thresholds: {mom_thresholds}")

    grid = []
    for f in fast_ls:
        for s in slow_ls:
            if s > f:  # ensure slow > fast
                for m in mom_periods:
                    if m >= s:  # ensure momentum period > slow period
                        for t in mom_thresholds:
                            grid.append([f, s, m, round(t, 5)])
    
    return grid

if __name__ == "__main__":
    df = load_df("round1", "2504071725_sma20_sma20_sma20")

    single_crossover_analyse(df, "SQUID_INK", 5, 10, 11, 0.0045)

    # analyse one
    # product = "SQUID_INK"
    # fast_l, slow_l = 15, 30
    # mom_period, mom_threshold = 31, 0.0035
    # single_crossover_analyse(df, "SQUID_INK", 
    #                          fast_l, slow_l,
    #                          mom_period, mom_threshold)

    # # usefull quick plotting of columns
    # plot_df_columns(
    # df,
    # columns=['mid_price', sma_name_10, sma_name_20],
    # products=products[2],
    # kind="line",
    # titles=None)    
   
    plt.show()
