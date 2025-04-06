import pandas as pd
import numpy as np
import re
import json


# Initialize a list to hold all parsed rows
# Load CSV file
file_path = r"C:\Users\jasschol\repositories\personal projects\total-drama-island\logs\tutorial\2504031627_market_maker\processed\2504031627_market_maker_sandbox.csv"  # Update with your actual file path
# 


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
df_final.to_csv("processed_logs.csv", index=False)

# Display the first few rows
print(df_final.head())

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the base path relative to the script's location
base_path = os.path.join(script_dir, '..', 'logs', 'tutorial')