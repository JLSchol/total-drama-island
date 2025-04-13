import os
import csv
import json
import pandas as pd
import re
import numpy as np

def split_all_logs(base_dir):
    """Finds and processes all .log files in base_dir and its subdirectories."""
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".log"):
                log_file_path = os.path.join(root, file)
                split_log(log_file_path)

def process_all_tradehistory(base_dir):
    """Finds and processes all .log files in base_dir and its subdirectories."""
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("tradehistory.txt"):
                trade_file_path = os.path.join(root, file)
                trade_history_txt_to_csv(trade_file_path)

def process_all_sandbox(base_dir):
    """Finds and processes all .log files in base_dir and its subdirectories."""
    for root, _, files in os.walk(base_dir):
        for file in files:
            print(f"{root=} --- {file=}")
            if file.endswith("sandbox.txt"):
                trade_file_path = os.path.join(root, file)
                sandbox_txt_to_csv(trade_file_path)

def get_all_printlogs_from_sandbox(base_dir):
    """Finds and processes all .log files in base_dir and its subdirectories."""
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("sandbox.csv"):
                sanbox_csv_file_path = os.path.join(root, file)
                printlogs_from_sandox_cxv(sanbox_csv_file_path, root, "printlogs.csv")

def split_log(log_file):
    """Processes a single log file and saves output in a 'processed' subdirectory."""
    with open(log_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sandbox_logs = []
    activities_data = []
    trade_history = []
    current_section = None

    for line in lines:
        line = line.strip()

        # Detect section headers
        if line.startswith("Sandbox logs:"):
            current_section = "sandbox"
            continue
        elif line.startswith("Activities log:"):
            current_section = "activities"
            continue
        elif line.startswith("Trade History:"):
            current_section = "trade_history"
            continue

        # Categorize log entries based on current section
        if current_section == "sandbox" and line:
            sandbox_logs.append(line)
        elif current_section == "activities" and line:
            activities_data.append(line)
        elif current_section == "trade_history" and line:
            trade_history.append(line)

    # Get original filename without extension
    base_name = os.path.splitext(os.path.basename(log_file))[0]

    # Define processed output directory
    log_dir = os.path.dirname(log_file)
    processed_dir = os.path.join(log_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)  # Create 'processed' directory if not exists

    # Define output file paths
    sandbox_file = os.path.join(processed_dir, f"{base_name}_sandbox.txt")
    activities_file = os.path.join(processed_dir, f"{base_name}_activities.csv")
    trade_history_file = os.path.join(processed_dir, f"{base_name}_tradehistory.txt")

    # Write Sandbox Logs to Text File 
    if sandbox_logs:
        with open(sandbox_file, "w", encoding="utf-8") as file:
            for log in sandbox_logs:
                file.write(log + "\n")

        with open(sandbox_file, 'r', encoding='utf-8') as file:
            # Load the content 
            content = file.read()    

        # Add brackets and comma's for valid json format
        content = f"[{content.strip().replace('}\n{', '},\n{')}]"
            
        with open(sandbox_file, 'w', encoding='utf-8') as file:
            file.write(content)

    # Write Activities Logs to CSV
    if activities_data:
        with open(activities_file, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            for row in activities_data:
                writer.writerow(row.split(";"))

    # Write Trade History to txt
    if trade_history:

        with open(trade_history_file, "w", encoding="utf-8") as file:
            for log in trade_history:
                file.write(log + "\n")


    print(f"Processed: {log_file} → {processed_dir}/")

def trade_history_txt_to_csv(input_file):
    # Step 1: Read the content from the input .txt file
    with open(input_file, 'r', encoding='utf-8') as file:
        # Load the content into a Python list (assuming the content is in JSON format)
        trade_history = json.loads(file.read())
    
    # Step 2: Generate the output file name with the same name as input file but with .csv extension
    output_file = os.path.splitext(input_file)[0] + '.csv'
    
    # Step 3: Write the data to the CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        # Define the header for the CSV file
        fieldnames = ['timestamp', 'buyer', 'seller', 'symbol', 'currency', 'price', 'quantity']
        
        # Create a CSV writer object
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Write all trade history rows to the CSV file
        writer.writerows(trade_history)

    print(f"Data has been successfully converted to {output_file}")

def sandbox_txt_to_csv(input_file):
    # Step 1: Read the content from the input .txt file
    with open(input_file, 'r', encoding='utf-8') as file:
        sand_box_logs = json.loads(file.read())

    # Step 2: Generate the output file name with the same name as input file but with .csv extension
    output_file = os.path.splitext(input_file)[0] + '.csv'
    
    # Step 3: Write the data to the CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        # Define the header for the CSV file
        fieldnames = ['sandboxLog', 'lambdaLog', 'timestamp']
        
        # Create a CSV writer object
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Write all trade history rows to the CSV file
        writer.writerows(sand_box_logs)

    print(f"Data has been successfully converted to {output_file}")

def printlogs_from_sandox_cxv(file_path, root, name):

    def extract_json_from_file(file_path):
        pattern = r'\{.*\}'  # Regex pattern to capture the JSON dictionary
        extracted_data = []

        with open(file_path, 'r', encoding='utf-8') as file:
            next(file)  # Skip the header line

            for line in file:
                match = re.search(pattern, line)
                if match:
                    json_string = match.group(0)  # Extract the JSON part
                    json_string = json_string.replace('""', '"')  # Fix double quotes

                    try:
                        json_dict = json.loads(json_string)  # Parse the JSON
                        extracted_data.append(json_dict)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line: {line}")
                        print(f"Error: {e}")

        return extracted_data

    def json_to_dataframe(json_list):
        """ Converts the extracted JSON list into a Pandas DataFrame """
        flattened_data = []

        for entry in json_list:
            for product, values in entry.items():
                # Add product name as a separate column
                row = {"product": product, **values}
                flattened_data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(flattened_data)

        # Convert "NaN" strings to actual NaN values
        df.replace("NaN", np.nan, inplace=True)

        return df

    # Example usage
    json_dict = extract_json_from_file(file_path)
    df_final = json_to_dataframe(json_dict)  # Convert to DataFrame

    # Save to CSV if needed
    printlogs_csv_file_path = os.path.join(root, name)
    df_final.to_csv(printlogs_csv_file_path, index=False)

if __name__ == "__main__":
    log_dir = os.path.dirname(__file__)
    # round_1_dir = os.path.join(log_dir, "round1")
    # round_2_dir = os.path.join(log_dir, "round2")
    # specific_dir = os.path.join(log_dir, "round2", "tryout_squid_tactics")
    specific_dir = os.path.join(log_dir, "round3", "test")
    split_all_logs(specific_dir)
    print("splitting complete")
    process_all_tradehistory(specific_dir)
    print("processing tradehistory complete")
    process_all_sandbox(specific_dir)
    print("processing all sandbox files complete")
    get_all_printlogs_from_sandbox(specific_dir)
    print("converted prints dicts from sanbox logs to data frame csv")

