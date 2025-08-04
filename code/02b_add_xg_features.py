import pandas as pd
import os

# Get the directory of this script and build paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

# Load the existing engineered features which now includes form data
engineered_df = pd.read_csv(os.path.join(data_dir, 'final_features_with_form.csv'))

# --- This is the real xG data you provided, structured into a Python list. ---
xg_data = [
    {'Team': 'Arsenal', 'Season': '2024/2025', 'xG': 59.9, 'xGA': 34.4},
    {'Team': 'Arsenal', 'Season': '2023/2024', 'xG': 76.1, 'xGA': 27.9},
    {'Team': 'Arsenal', 'Season': '2022/2023', 'xG': 71.6, 'xGA': 42.0},
    {'Team': 'Arsenal', 'Season': '2021/2022', 'xG': 60.5, 'xGA': 45.7},
    {'Team': 'Arsenal', 'Season': '2020/2021', 'xG': 51.7, 'xGA': 43.0},
    {'Team': 'Man City', 'Season': '2024/2025', 'xG': 68.1, 'xGA': 47.7},
    {'Team': 'Man City', 'Season': '2023/2024', 'xG': 80.5, 'xGA': 35.6},
    {'Team': 'Man City', 'Season': '2022/2023', 'xG': 78.6, 'xGA': 32.1},
    {'Team': 'Man City', 'Season': '2021/2022', 'xG': 88.7, 'xGA': 24.6},
    {'Team': 'Man City', 'Season': '2020/2021', 'xG': 68.2, 'xGA': 30.2},
    {'Team': 'Liverpool', 'Season': '2024/2025', 'xG': 82.2, 'xGA': 38.6},
    {'Team': 'Liverpool', 'Season': '2023/2024', 'xG': 87.8, 'xGA': 45.7},
    {'Team': 'Liverpool', 'Season': '2022/2023', 'xG': 71.5, 'xGA': 50.8},
    {'Team': 'Liverpool', 'Season': '2021/2022', 'xG': 88.7, 'xGA': 33.8},
    {'Team': 'Liverpool', 'Season': '2020/2021', 'xG': 67.5, 'xGA': 43.0},
    {'Team': 'Man United', 'Season': '2024/2025', 'xG': 52.6, 'xGA': 53.8},
    {'Team': 'Man United', 'Season': '2023/2024', 'xG': 56.5, 'xGA': 68.9},
    {'Team': 'Man United', 'Season': '2022/2023', 'xG': 67.7, 'xGA': 50.4},
    {'Team': 'Man United', 'Season': '2021/2022', 'xG': 55.8, 'xGA': 53.0},
    {'Team': 'Man United', 'Season': '2020/2021', 'xG': 60.1, 'xGA': 41.4},
    {'Team': 'Chelsea', 'Season': '2024/2025', 'xG': 67.8, 'xGA': 47.3},
    {'Team': 'Chelsea', 'Season': '2023/2024', 'xG': 74.5, 'xGA': 58.1},
    {'Team': 'Chelsea', 'Season': '2022/2023', 'xG': 49.5, 'xGA': 52.5},
    {'Team': 'Chelsea', 'Season': '2021/2022', 'xG': 63.4, 'xGA': 33.2},
    {'Team': 'Chelsea', 'Season': '2020/2021', 'xG': 62.4, 'xGA': 30.3},
    {'Team': 'Tottenham', 'Season': '2024/2025', 'xG': 58.8, 'xGA': 63.3},
    {'Team': 'Tottenham', 'Season': '2023/2024', 'xG': 68.2, 'xGA': 53.4},
    {'Team': 'Tottenham', 'Season': '2022/2023', 'xG': 57.0, 'xGA': 49.6},
    {'Team': 'Tottenham', 'Season': '2021/2022', 'xG': 61.2, 'xGA': 39.3},
    {'Team': 'Tottenham', 'Season': '2020/2021', 'xG': 53.1, 'xGA': 49.1},
    {'Team': 'Aston Villa', 'Season': '2024/2025', 'xG': 56.1, 'xGA': 50.1},
    {'Team': 'Aston Villa', 'Season': '2023/2024', 'xG': 63.3, 'xGA': 59.9},
    {'Team': 'Aston Villa', 'Season': '2022/2023', 'xG': 50.3, 'xGA': 52.5},
    {'Team': 'Aston Villa', 'Season': '2021/2022', 'xG': 44.0, 'xGA': 49.0},
    {'Team': 'Aston Villa', 'Season': '2020/2021', 'xG': 52.5, 'xGA': 51.1},
    {'Team': 'Newcastle', 'Season': '2024/2025', 'xG': 63.8, 'xGA': 45.5},
    {'Team': 'Newcastle', 'Season': '2023/2024', 'xG': 76.0, 'xGA': 61.4},
    {'Team': 'Newcastle', 'Season': '2022/2023', 'xG': 71.9, 'xGA': 39.5},
    {'Team': 'Newcastle', 'Season': '2021/2022', 'xG': 38.1, 'xGA': 57.1},
    {'Team': 'Newcastle', 'Season': '2020/2021', 'xG': 43.4, 'xGA': 58.3},
    {'Team': 'Brighton', 'Season': '2024/2025', 'xG': 58.7, 'xGA': 54.6},
    {'Team': 'Brighton', 'Season': '2023/2024', 'xG': 56.8, 'xGA': 55.4},
    {'Team': 'Brighton', 'Season': '2022/2023', 'xG': 73.3, 'xGA': 50.2},
    {'Team': 'Brighton', 'Season': '2021/2022', 'xG': 46.2, 'xGA': 42.9},
    {'Team': 'Brighton', 'Season': '2020/2021', 'xG': 50.9, 'xGA': 35.3},
    {'Team': 'West Ham', 'Season': '2024/2025', 'xG': 47.0, 'xGA': 59.7},
    {'Team': 'West Ham', 'Season': '2023/2024', 'xG': 52.3, 'xGA': 71.1},
    {'Team': 'West Ham', 'Season': '2022/2023', 'xG': 49.2, 'xGA': 53.0},
    {'Team': 'West Ham', 'Season': '2021/2022', 'xG': 51.4, 'xGA': 53.5},
    {'Team': 'West Ham', 'Season': '2020/2021', 'xG': 55.4, 'xGA': 48.7},
    {'Team': 'Wolves', 'Season': '2024/2025', 'xG': 43.7, 'xGA': 58.1},
    {'Team': 'Wolves', 'Season': '2023/2024', 'xG': 46.7, 'xGA': 67.7},
    {'Team': 'Wolves', 'Season': '2022/2023', 'xG': 36.8, 'xGA': 59.9},
    {'Team': 'Wolves', 'Season': '2021/2022', 'xG': 37.5, 'xGA': 56.9},
    {'Team': 'Wolves', 'Season': '2020/2021', 'xG': 36.5, 'xGA': 49.5},
    {'Team': 'Crystal Palace', 'Season': '2024/2025', 'xG': 60.4, 'xGA': 49.1},
    {'Team': 'Crystal Palace', 'Season': '2023/2024', 'xG': 48.6, 'xGA': 52.0},
    {'Team': 'Crystal Palace', 'Season': '2022/2023', 'xG': 39.3, 'xGA': 48.1},
    {'Team': 'Crystal Palace', 'Season': '2021/2022', 'xG': 46.4, 'xGA': 40.7},
    {'Team': 'Crystal Palace', 'Season': '2020/2021', 'xG': 34.1, 'xGA': 58.2},
    {'Team': 'Bournemouth', 'Season': '2024/2025', 'xG': 64.0, 'xGA': 48.5},
    {'Team': 'Bournemouth', 'Season': '2023/2024', 'xG': 55.9, 'xGA': 58.1},
    {'Team': 'Bournemouth', 'Season': '2022/2023', 'xG': 38.5, 'xGA': 63.8},
    {'Team': 'Bournemouth', 'Season': '2021/2022', 'xG': 75.0, 'xGA': 46.4},
    {'Team': 'Bournemouth', 'Season': '2020/2021', 'xG': 64.4, 'xGA': 50.0},
    {'Team': 'Brentford', 'Season': '2024/2025', 'xG': 59.0, 'xGA': 55.4},
    {'Team': 'Brentford', 'Season': '2023/2024', 'xG': 58.2, 'xGA': 56.0},
    {'Team': 'Brentford', 'Season': '2022/2023', 'xG': 56.3, 'xGA': 48.8},
    {'Team': 'Brentford', 'Season': '2021/2022', 'xG': 45.8, 'xGA': 48.5},
    {'Team': 'Brentford', 'Season': '2020/2021', 'xG': 74.9, 'xGA': 39.4},
    {'Team': 'Fulham', 'Season': '2024/2025', 'xG': 49.0, 'xGA': 47.2},
    {'Team': 'Fulham', 'Season': '2023/2024', 'xG': 50.8, 'xGA': 62.9},
    {'Team': 'Fulham', 'Season': '2022/2023', 'xG': 46.2, 'xGA': 63.6},
    {'Team': 'Fulham', 'Season': '2021/2022', 'xG': 95.1, 'xGA': 43.3},
    {'Team': 'Fulham', 'Season': '2020/2021', 'xG': 40.5, 'xGA': 52.6},
    {'Team': 'Everton', 'Season': '2024/2025', 'xG': 41.8, 'xGA': 46.2},
    {'Team': 'Everton', 'Season': '2023/2024', 'xG': 54.0, 'xGA': 55.2},
    {'Team': 'Everton', 'Season': '2022/2023', 'xG': 45.2, 'xGA': 65.5},
    {'Team': 'Everton', 'Season': '2021/2022', 'xG': 41.2, 'xGA': 55.4},
    {'Team': 'Everton', 'Season': '2020/2021', 'xG': 45.7, 'xGA': 50.1},
    {'Team': 'Nott\'m Forest', 'Season': '2024/2025', 'xG': 45.5, 'xGA': 48.9},
    {'Team': 'Nott\'m Forest', 'Season': '2023/2024', 'xG': 49.9, 'xGA': 53.3},
    {'Team': 'Nott\'m Forest', 'Season': '2022/2023', 'xG': 39.3, 'xGA': 64.2},
    {'Team': 'Nott\'m Forest', 'Season': '2021/2022', 'xG': 68.6, 'xGA': 54.3},
    {'Team': 'Nott\'m Forest', 'Season': '2020/2021', 'xG': 49.8, 'xGA': 52.2},
    {'Team': 'Burnley', 'Season': '2024/2025', 'xG': 57.5, 'xGA': 39.1},
    {'Team': 'Burnley', 'Season': '2023/2024', 'xG': 40.6, 'xGA': 70.4},
    {'Team': 'Burnley', 'Season': '2022/2023', 'xG': 66.2, 'xGA': 38.2},
    {'Team': 'Burnley', 'Season': '2021/2022', 'xG': 39.7, 'xGA': 57.1},
    {'Team': 'Burnley', 'Season': '2020/2021', 'xG': 39.3, 'xGA': 54.7},
    {'Team': 'Leeds', 'Season': '2024/2025', 'xG': 89.1, 'xGA': 29.6},
    {'Team': 'Leeds', 'Season': '2023/2024', 'xG': 79.5, 'xGA': 38.0},
    {'Team': 'Leeds', 'Season': '2022/2023', 'xG': 47.3, 'xGA': 67.1},
    {'Team': 'Leeds', 'Season': '2021/2022', 'xG': 44.4, 'xGA': 67.8},
    {'Team': 'Leeds', 'Season': '2020/2021', 'xG': 55.6, 'xGA': 57.9},
    {'Team': 'Sunderland', 'Season': '2024/2025', 'xG': 58.1, 'xGA': 49.0},
    {'Team': 'Sunderland', 'Season': '2023/2024', 'xG': 61.7, 'xGA': 50.5},
    {'Team': 'Sunderland', 'Season': '2022/2023', 'xG': 58.3, 'xGA': 52.3},
    {'Team': 'Sunderland', 'Season': '2021/2022', 'xG': 79.1, 'xGA': 52.9},
    {'Team': 'Sunderland', 'Season': '2020/2021', 'xG': 71.5, 'xGA': 43.6},
]

# Convert to DataFrame
xg_df = pd.DataFrame(xg_data)

# --- Step 1: Merge the xG data with our engineered features ---
# We use 'left' to ensure we keep all teams from our main engineered_df
merged_df = pd.merge(engineered_df, xg_df, on=['Team', 'Season'], how='left')

# --- Step 2: Create new, more powerful features from xG data ---
merged_df['xG_diff'] = merged_df['xG'] - merged_df['xGA']

# --- Step 3: Create lagged xG features (previous season's xG) ---
merged_df['prev_season_xG'] = merged_df.groupby('Team')['xG'].shift(1)
merged_df['prev_season_xGA'] = merged_df.groupby('Team')['xGA'].shift(1)
merged_df['prev_season_xG_diff'] = merged_df.groupby('Team')['xG_diff'].shift(1)

# --- Step 4: Clean up NaNs for the first season's data ---
merged_df = merged_df.fillna(0)

# --- Step 5: Save the new, richer dataset ---
merged_df.to_csv(os.path.join(data_dir, 'final_features_with_xg.csv'), index=False)

print(f"Engineered features with real xG data created and saved to `{os.path.join(data_dir, 'final_features_with_xg.csv')}`")
