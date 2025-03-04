import os
import csv

def process_log_file(log_file):
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

    # Write Sandbox Logs to Text File (or CSV with key-value pairs)
    if sandbox_logs:

        with open(sandbox_file, "w", encoding="utf-8") as file:
            for log in sandbox_logs:
                file.write(log + "\n")

    # Write Activities Logs to CSV
    if activities_data:
        with open(activities_file, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            for row in activities_data:
                writer.writerow(row.split(";"))

    # Write Trade History to CSV
    if trade_history:

        with open(trade_history_file, "w", encoding="utf-8") as file:
            for log in trade_history:
                file.write(log + "\n")


    print(f"Processed: {log_file} → {processed_dir}/")


def process_all_logs(base_dir):
    """Finds and processes all .log files in base_dir and its subdirectories."""
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".log"):
                log_file_path = os.path.join(root, file)
                process_log_file(log_file_path)


if __name__ == "__main__":
    logs_directory = os.path.join(os.path.dirname(__file__), "logs")
    process_all_logs(logs_directory)
    print("Processing complete.")
