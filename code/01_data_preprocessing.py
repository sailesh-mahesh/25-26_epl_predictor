import pandas as pd
import os

# Define the paths to your data folders
pl_path = 'data/premier_league/'
championship_path = 'data/championship/'

def load_data(folder_path, league_name):
    """
    Loads all CSV files from a given folder, combines them,
    and adds a 'League' column.
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    all_data = []
    for file in all_files:
        df = pd.read_csv(file, encoding='latin1')
        df['League'] = league_name
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

# Load Premier League and Championship data
pl_df = load_data(pl_path, 'Premier League')
championship_df = load_data(championship_path, 'Championship')

# Combine both DataFrames into a single master DataFrame
master_df = pd.concat([pl_df, championship_df], ignore_index=True)

# Clean the data by dropping empty columns
master_df = master_df.dropna(axis=1, how='all')

# Let's inspect the combined and cleaned data
print("Shape of the combined dataframe:", master_df.shape)
print("\nFirst 5 rows:")
print(master_df.head())
print("\nColumns:")
print(master_df.columns)

# Save the combined dataframe to a new CSV for later use
master_df.to_csv('data/combined_data.csv', index=False)
print("\nCombined data saved to `data/combined_data.csv`")