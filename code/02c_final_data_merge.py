import pandas as pd
import os

# Get the directory of this script and build paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

# Load the engineered features with form data
form_df = pd.read_csv(os.path.join(data_dir, 'final_features_with_form.csv'))

# Load the real xG data you provided
xg_df = pd.read_csv(os.path.join(data_dir, 'xg_data.csv'))

# Merge the two dataframes on 'Team' and 'Season'
# We use a left merge to ensure we keep all teams from the form_df
final_df = pd.merge(form_df, xg_df, on=['Team', 'Season'], how='left')

# Create new features from the xG data
final_df['xG_diff'] = final_df['xG'] - final_df['xGA']
final_df['prev_season_xG'] = final_df.groupby('Team')['xG'].shift(1)
final_df['prev_season_xGA'] = final_df.groupby('Team')['xGA'].shift(1)
final_df['prev_season_xG_diff'] = final_df.groupby('Team')['xG_diff'].shift(1)

# Clean up any NaNs that may have been created during the merge
final_df = final_df.fillna(0)

# Save the final, complete dataset
final_df.to_csv(os.path.join(data_dir, 'final_features_complete.csv'), index=False)

print(f"Final complete dataset saved to `{os.path.join(data_dir, 'final_features_complete.csv')}`")
print("\nSample of the final dataframe:")
print(final_df[['Team', 'Season', 'Points', 'Form Points Last 10', 'xG_diff', 'prev_season_xG_diff']].tail())