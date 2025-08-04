import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# Get the directory of this script and build paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

# --- Step 1: Load the final engineered data with all features ---
df = pd.read_csv(os.path.join(data_dir, 'final_features_complete.csv'))

# --- Step 2: Define features (X) and target (y) ---
# Add a transfer impact column and fill with 0s for historical data
df['transfer_impact'] = 0

# --- FIX: Calculate 'adjusted_prev_points' for ALL historical data with a higher IMPACT_FACTOR ---
IMPACT_FACTOR = 6.0 # SIGNIFICANTLY INCREASED IMPACT FACTOR for stronger effect
df['adjusted_prev_points'] = df['prev_season_points'] + (df['transfer_impact'] * IMPACT_FACTOR)

features = [
    'prev_season_gd', 
    'promoted_from_championship',
    'prev_season_xG_diff',
    'prev_season_form',
    'prev_pl_avg_points', 
    'adjusted_prev_points' # This is the primary point feature now
]

target_points = 'Points'
target_goals_scored = 'Goals Scored'
target_goals_conceded = 'Goals Conceded'

# --- Step 3: Scale the transfer impact feature on a consistent range ---
scaler = MinMaxScaler()
scaler.fit(np.array([-10, 10]).reshape(-1, 1))

# Apply the scaling to the historical data's 'transfer_impact' column
# This is already implicitly handled within adjusted_prev_points.

# --- Step 4: Split the data into training and prediction sets ---
df['promoted_from_championship'] = df['promoted_from_championship'].astype(int)

train_data = df[(df['League'] == 'Premier League') & (df['Season'] != '2024/2025')].copy()
X_train = train_data[features]
y_train_points = train_data[target_points]
y_train_goals_scored = train_data[target_goals_scored]
y_train_goals_conceded = train_data[target_goals_conceded]

# --- Step 5: Train three Random Forest Regression models with optimized hyperparameters ---
model_points = RandomForestRegressor(n_estimators=1000, max_depth=25, min_samples_leaf=1, random_state=42)
model_goals_scored = RandomForestRegressor(n_estimators=1000, max_depth=25, min_samples_leaf=1, random_state=42)
model_goals_conceded = RandomForestRegressor(n_estimators=1000, max_depth=25, min_samples_leaf=1, random_state=42)

model_points.fit(X_train, y_train_points)
model_goals_scored.fit(X_train, y_train_goals_scored)
model_goals_conceded.fit(X_train, y_train_goals_conceded)

# --- Step 6: Prepare data for 2025/2026 predictions with interactive input ---
pl_25_26_teams = [
    'Arsenal', 'Man City', 'Liverpool', 'Man United', 'Chelsea', 'Tottenham', 'Aston Villa',
    'Newcastle', 'West Ham', 'Crystal Palace', 'Brighton', 'Fulham', 'Wolves', 'Everton',
    'Brentford', 'Bournemouth', "Nott'm Forest",
    'Leeds', 'Burnley', 'Sunderland'
]

prediction_df = df[df['Season'] == '2024/2025'].copy()
prediction_df = prediction_df[prediction_df['Team'].isin(pl_25_26_teams)].copy()
prediction_df['promoted_from_championship'] = prediction_df['League'].apply(lambda x: 1 if x == 'Championship' else 0)

print("\n--- Interactive Transfer Impact Input ---")
print("Enter a numerical impact for each team's transfers.")
print("Example: 5 for a major signing, -5 for losing a star player, 0 for no change.")
print("Press Enter to use a default of 0.")

transfer_impacts = {}
for team in pl_25_26_teams:
    while True:
        try:
            impact_input = input(f"Enter transfer impact for {team}: ")
            if impact_input == '':
                impact_value = 0
            else:
                impact_value = float(impact_input)
            transfer_impacts[team] = impact_value
            break
        except ValueError:
            print("Invalid input. Please enter a number or press Enter for 0.")

prediction_df['transfer_impact'] = prediction_df['Team'].map(transfer_impacts).fillna(0)

# --- Apply transfer impact to create 'adjusted_prev_points' for prediction ---
prediction_df['adjusted_prev_points'] = prediction_df['prev_season_points'] + (prediction_df['transfer_impact'] * IMPACT_FACTOR)

# --- Step 8: Make the 2025/2026 predictions using all three models ---
X_predict = prediction_df[features]
predicted_points = model_points.predict(X_predict)
predicted_goals_scored = model_goals_scored.predict(X_predict)
predicted_goals_conceded = model_goals_conceded.predict(X_predict)

prediction_df['Predicted Points'] = predicted_points.round(0)
prediction_df['Predicted GF'] = predicted_goals_scored.round(0)
prediction_df['Predicted GA'] = predicted_goals_conceded.round(0)
prediction_df['Predicted GD'] = prediction_df['Predicted GF'] - prediction_df['Predicted GA']

# Generate the final league table
final_table = prediction_df[['Team', 'Predicted Points', 'Predicted GF', 'Predicted GA', 'Predicted GD']].sort_values(by=['Predicted Points', 'Predicted GD'], ascending=[False, False])
final_table = final_table.reset_index(drop=True)
final_table.index = final_table.index + 1
final_table.index.name = 'Position'

print("\n\nPredicted 2025/2026 Premier League Table (with all features):")
print(final_table)