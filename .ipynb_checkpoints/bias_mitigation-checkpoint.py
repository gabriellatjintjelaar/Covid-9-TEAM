import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import f_oneway

# Load dataset
df = pd.read_csv('processed_data/100k_population_data.csv', low_memory=False)

# Exclude territories
territories_to_exclude = ['US_GU', 'US_VI', 'US_AS', 'US_PR', 'US_MP', 'US_DC']
df = df[~df['location_key'].isin(territories_to_exclude)]

# Define state lists
blue_states_list = ['US_CA', 'US_ME', 'US_OR', 'US_CO', 'US_MD', 'US_RI', 'US_CT', 'US_MA', 'US_VT', 'US_DE', 'US_NH', 'US_VA', 'US_NJ', 'US_WA', 'US_HI', 'US_NM', 'US_NY', 'US_IL']
red_states_list = ['US_AL', 'US_AK', 'US_AR', 'US_ID', 'US_IN', 'US_IA', 'US_KS', 'US_KY', 'US_LA', 'US_MS', 'US_MO', 'US_MT', 'US_NE', 'US_ND', 'US_OK', 'US_SC', 'US_SD', 'US_TN', 'US_TX', 'US_UT', 'US_WV', 'US_WY']
swing_states_list = ['US_AZ', 'US_NV', 'US_FL', 'US_NC', 'US_GA', 'US_OH', 'US_MI', 'US_PA', 'US_MN', 'US_WI']

# Create dummy variables for state types
df['blue_states'] = df['location_key'].apply(lambda x: 1 if x in blue_states_list else 0)
df['red_states'] = df['location_key'].apply(lambda x: 1 if x in red_states_list else 0)
df['swing_states'] = df['location_key'].apply(lambda x: 1 if x in swing_states_list else 0)

# Create a state_type column
def get_state_type(row):
    if row['blue_states'] == 1:
        return 'Blue'
    elif row['red_states'] == 1:
        return 'Red'
    elif row['swing_states'] == 1:
        return 'Swing'
    else:
        return 'Other'

df['state_type'] = df.apply(get_state_type, axis=1)

# Drop unneeded columns
columns_to_drop = ['blue_states', 'red_states', 'swing_states', 'target_end_date', 'location_key', 'location', 'new_hospitalized_patients', 'hospitalized_per_100k', 'unemployment_rate', 'year']

# Drop the specified columns
df = df.drop(columns=columns_to_drop)

# Displat dataframe
display(df)

# Define strata
strata = df['state_type'].unique()

# Perform stratified sampling
sample_df = pd.DataFrame(columns=df.columns)

for stratum in strata:
    stratum_df = df[df['state_type'] == stratum]
    stratum_size = int(0.8 * len(stratum_df))  # Adjust the sample size as needed
    stratum_sample = stratum_df.sample(n=stratum_size, random_state=42)
    sample_df = pd.concat([sample_df, stratum_sample])

# Display the stratifies sample
print("Stratified sample:")
display(sample_df)

# Define encode_state_type function
def encode_state_type(state):
    if state == 'Blue':
        return 0
    elif state == 'Red':
        return 1
    else: 
        return 2 # For Swing states

sample_df['state_type'] = sample_df['state_type'].apply(encode_state_type)

# Split data into features (X) and target variable (y)
X = sample_df.drop(columns=['cases_per_100k', 'total_population', 'inc cases'])
y = sample_df['cases_per_100k']  

# Training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Cross-validation
cv = KFold(n_splits=5, random_state=42, shuffle=True)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

# Print evaluation metrics
print(f'R2: {r2:.4f}')
print(f'Test RMSE: {rmse:.4f}')
print(f'Cross-Validation RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std():.4f})')

# Create DataFrame with actual and predicted values
y_test_df = pd.DataFrame({
    'cases_per_100k': y_test,  # Actual values
    'state_type': X_test['state_type'],  # Encoded state type
    'predicted': y_pred  # Predicted values
})

# Define a function to calculate RMSE
def calculate_rmse(group):
    return np.sqrt(mean_squared_error(group['cases_per_100k'], group['predicted']))

# Group by state type and calculate RMSE for each group
rmse_by_state_type = y_test_df.groupby('state_type').apply(calculate_rmse).reset_index()
rmse_by_state_type.columns = ['State Type', 'RMSE']

# AIC calculation
n = len(y_test)
rss = np.sum((y_test - y_pred) ** 2)
k_rf = len(rf_model.estimators_) + 1
aic_rf = n * np.log(rss / n) + 2 * k_rf
print(f'Random Forest AIC: {aic_rf:.4f}')

# Additional model evaluation metrics
test_mae = mean_absolute_error(y_test, y_pred)
print(f'Test MAE: {test_mae:.4f}')

# ANOVA F-value and p-value
f_value, p_value = f_oneway(sample_df[sample_df['state_type'] == 0]['cases_per_100k'],
                             sample_df[sample_df['state_type'] == 1]['cases_per_100k'],
                             sample_df[sample_df['state_type'] == 2]['cases_per_100k'])
print("ANOVA F-value:", f_value)
print("p-value:", p_value)

# Demographic Parity
demographic_parity = sample_df.groupby('state_type')['cases_per_100k'].mean().reset_index()
demographic_parity.columns = ['State Type', 'Mean Predicted Value']
print("\nDemographic Parity:\n", demographic_parity)

# Equalized Odds (Residuals)
# Calculate residuals
residuals = y_test - y_pred

# Create a DataFrame with residuals and state_type
results_df = pd.DataFrame({'Residuals': residuals, 'state_type': X_test['state_type']})

# Group by state_type and calculate mean and standard deviation of residuals
residuals_by_state_type = results_df.groupby('state_type').agg({'Residuals': [list, np.mean, np.std]}).reset_index()

# Rename columns for clarity
residuals_by_state_type.columns = ['State_Type', 'Residuals', 'Mean Residual', 'Std Residual']

print("\nEqualized Odds (Residuals):")
print(residuals_by_state_type[['State_Type', 'Mean Residual', 'Std Residual']])

# Predictive Parity (Mean Absolute Error)
mae_by_state_type = y_test_df.groupby('state_type').apply(lambda x: mean_absolute_error(x['cases_per_100k'], x['predicted'])).reset_index()
mae_by_state_type.columns = ['State Type', 'MAE']
print("\nPredictive Parity (Mean Absolute Error):")
print(mae_by_state_type)

# Data Points Count by State Type
data_points_count = sample_df['state_type'].value_counts().reset_index()
data_points_count.columns = ['State Type', 'Counts']
print("\nData Points Count by State Type:\n", data_points_count)
