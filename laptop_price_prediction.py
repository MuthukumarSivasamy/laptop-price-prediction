

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv(r"C:\Users\muthu\Downloads\laptop.csv")

df = df[['Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price' ]]
df = df.dropna()
df = df.drop_duplicates()

dup_col = ['Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight' ]
df = df.groupby(dup_col, as_index=False)['Price'].mean()

df['Inches'] = df['Inches'].replace('?',np.nan)
df['Inches'] = df['Inches'].fillna(df['Inches'].mode()[0])
df['Inches'] = df['Inches'].astype(float)

lower = df['Inches'].quantile(0.01)
upper = df['Inches'].quantile(0.99)

df['Inches'] = np.where(df['Inches'] < lower, lower, df['Inches'])
df['Inches'] = np.where(df['Inches'] > upper, upper, df['Inches'])


df.rename(columns={'Weight': 'Weight_in_Kgs'},inplace=True)
df['Weight_in_Kgs'] = df['Weight_in_Kgs'].astype(str).str.replace('kg', '', case=False).str.strip()

df['Weight_in_Kgs'] = pd.to_numeric(df['Weight_in_Kgs'], errors='coerce')
df['Weight_in_Kgs'] = df['Weight_in_Kgs'].fillna(df['Weight_in_Kgs'].median())

lower = df['Weight_in_Kgs'].quantile(0.01)
upper = df['Weight_in_Kgs'].quantile(0.99)

df['Weight_in_Kgs'] = np.where(df['Weight_in_Kgs'] < lower, lower, df['Weight_in_Kgs'])
df['Weight_in_Kgs'] = np.where(df['Weight_in_Kgs'] > upper, upper, df['Weight_in_Kgs'])


pattern = r'[?@#$%&*!^~]'
mask = df.applymap(lambda x: bool(re.search(pattern, str(x))))
df[mask.any(axis=1)]


df['Memory'] = df['Memory'].replace('?',np.nan)
df['Memory'] = df['Memory'].fillna(df['Memory'].mode()[0])

df.rename(columns={'Ram':'Ram_GB'},inplace=True)
df['Ram_GB'] = df['Ram_GB'].astype(str).str.replace('GB', '', case=False).str.strip()
df['Ram_GB'] = pd.to_numeric(df['Ram_GB'], errors='coerce')

# Touchscreen (check if 'Touchscreen' present)
df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False).astype(int)

# IPS Panel
df['IPS'] = df['ScreenResolution'].str.contains('IPS Panel', case=False).astype(int)

# Extract resolution like "1920x1080"
df['Resolution'] = df['ScreenResolution'].str.extract(r'(\d+x\d+)')   # d for digit

# Split into X_res and Y_res
df[['X_res', 'Y_res']] = df['Resolution'].str.split('x', expand=True).astype(float)   # expand=True gives the dataframe format(normally dataseries format [1920, 1080])

df['PPI'] = ((df['X_res']**2 + df['Y_res']**2)**0.5) / df['Inches']

df.drop(['ScreenResolution', 'Resolution'], axis=1, inplace=True)

# 1. Extract Brand (Intel, AMD, Samsung, etc.)
df['Cpu_Brand'] = df['Cpu'].str.split().str[0]

# 2. Extract CPU Type (Core i5, Core i7, Ryzen, Celeron, etc.)
def extract_cpu_type(text):
    if 'Intel Core i3' in text:
        return 'Intel Core i3'
    elif 'Intel Core i5' in text:
        return 'Intel Core i5'
    elif 'Intel Core i7' in text:
        return 'Intel Core i7'
    elif 'Intel Core M' in text:
        return 'Intel Core M'
    elif 'Intel Celeron' in text:
        return 'Intel Celeron'
    elif 'Intel Pentium' in text:
        return 'Intel Pentium'
    elif 'Intel Xeon' in text:
        return 'Intel Xeon'
    elif 'AMD Ryzen' in text:
        return 'AMD Ryzen'
    elif 'AMD A4' in text:
        return 'AMD A4'
    elif 'AMD A6' in text:
        return 'AMD A6'
    elif 'AMD A8' in text:
        return 'AMD A8'
    elif 'AMD A9' in text:
        return 'AMD A9'
    elif 'AMD A10' in text:
        return 'AMD A10'
    elif 'AMD A12' in text:
        return 'AMD A12'
    elif 'AMD FX' in text:
        return 'AMD FX'
    elif 'AMD E-Series' in text:
        return 'AMD E-Series'
    else:
        return 'Other'

df['Cpu_Type'] = df['Cpu'].apply(extract_cpu_type)

df['Cpu_Speed'] = df['Cpu'].str.extract(r'(\d+(?:\.\d+)?)\s*GHz', flags=re.IGNORECASE)[0].astype(float)

df.drop('Cpu',axis=1,inplace=True)
df['Memory'] = df['Memory'].str.replace('GB', '', regex=False)
df['Memory'] = df['Memory'].str.replace('TB', '000', regex=False)
df['Memory'] = df['Memory'].str.replace('.', '', regex=False)  

# Create blank columns
df['HDD_Size_GB'] = 0
df['SSD_Size_GB'] = 0
df['Flash_Storage_GB'] = 0
df['Hybrid_Storage_GB'] = 0

'''
# Function to extract size
def extract_memory(row):
    for part in row.split('+'):
        part = part.strip()
        if 'HDD' in part:
            df.loc[df['Memory'] == row, 'HDD_Size_GB'] += int(part.split()[0])
        elif 'SSD' in part:
            df.loc[df['Memory'] == row, 'SSD_Size_GB'] += int(part.split()[0])
        elif 'Flash Storage' in part:
            df.loc[df['Memory'] == row, 'Flash_Storage_GB'] += int(part.split()[0])
        elif 'Hybrid' in part:
            df.loc[df['Memory'] == row, 'Hybrid_Storage_GB'] += int(part.split()[0])

df['Memory'].dropna().unique()  # Avoid nulls
for val in df['Memory'].dropna().unique():
    extract_memory(val)
'''
# Memory type columns
def extract_memory_apply(row):
    hdd = ssd = flash = hybrid = 0
    for part in row.split('+'):
        part = part.strip()
        if 'HDD' in part:
            hdd += int(part.split()[0])
        elif 'SSD' in part:
            ssd += int(part.split()[0])
        elif 'Flash Storage' in part:
            flash += int(part.split()[0])
        elif 'Hybrid' in part:
            hybrid += int(part.split()[0])
    return pd.Series([hdd, ssd, flash, hybrid])

df[['HDD_Size_GB', 'SSD_Size_GB', 'Flash_Storage_GB', 'Hybrid_Storage_GB']] = df['Memory'].apply(extract_memory_apply)
df.drop('Memory', axis=1, inplace=True)


df['Gpu_Brand'] = df['Gpu'].str.split().str[0]

#df['Is_Dedicated_GPU'] = df['Gpu_Brand'].apply(lambda x: 0 if x == 'Intel' else 1)

def classify_gpu(gpu_name):
    gpu_name = gpu_name.lower()
    if 'intel' in gpu_name:
        return 0
    elif any(x in gpu_name for x in ['radeon r2', 'radeon r3', 'radeon r4']) or \
         ('radeon r5' in gpu_name and 'm' not in gpu_name) or \
         'vega' in gpu_name:
        return 0
    else:
        return 1
    
df['Is_Dedicated_GPU'] = df['Gpu_Brand'].apply(classify_gpu)

# df['Gpu_Model'] = df['Gpu'].str.extract(r'(GTX|MX|HD|UHD|Quadro|Radeon|Iris|FirePro|Mali)', expand=False)
df['Gpu_Model'] = df['Gpu'].str.extract(
    r'(GTX|RTX|MX|GT|RX|HD|UHD|Iris\s?Plus|Iris\s?Pro|Iris|Quadro|FirePro|Radeon\sPro|Radeon|Mali|GeForce)',
    expand=False
)

df.drop('Gpu',axis=1,inplace=True)

#df['Weight_in_Kgs'] = df['Weight_in_Kgs'].replace(df['Weight_in_Kgs'].min(), df['Weight_in_Kgs'].mode()[0])

df = df.drop(['X_res', 'Y_res'],axis=1)

# Correlation matrix (only for numerical features)
plt.figure(figsize=(10,4))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot= True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# encoding

cat_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical Columns:", cat_cols)

label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # store the encoder if you want to inverse later


"""
# Define thresholds
low_thresh = df['Price'].quantile(0.33)
high_thresh = df['Price'].quantile(0.66)

# Segment the data
df_budget = df[df['Price'] <= low_thresh]
df_mid = df[(df['Price'] > low_thresh) & (df['Price'] <= high_thresh)]
df_premium = df[df['Price'] > high_thresh]

df['Log_Price'] = np.log1p(df['Price'])

features = [col for col in df.columns if col not in ['Price', 'Log_Price']]

X_budget = df_budget[features]
y_budget = np.log1p(df_budget['Price'])

X_mid = df_mid[features]
y_mid = np.log1p(df_mid['Price'])

X_premium = df_premium[features]
y_premium = np.log1p(df_premium['Price'])





# Budget segment
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_budget, y_budget, test_size=0.3, random_state=42
)

# Mid-range segment
Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_mid, y_mid, test_size=0.3, random_state=42
)

# Premium segment
Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_premium, y_premium, test_size=0.3, random_state=42
)


'''
model_budget = RandomForestRegressor(random_state=42)
model_budget.fit(X_budget, y_budget)

model_mid = RandomForestRegressor(random_state=42)
model_mid.fit(X_mid, y_mid)

model_premium = RandomForestRegressor(random_state=42)
model_premium.fit(X_premium, y_premium)



'''
model_budget = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_budget.fit(X_budget, y_budget)

model_mid = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_mid.fit(X_mid, y_mid)

model_premium = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_premium.fit(X_premium, y_premium)



def evaluate_model(model, X_test, y_test, segment_name):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    r2 = r2_score(y_test_actual, y_pred)
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

    print(f"📊 {segment_name} Laptops Performance:")
    print(f"R² Score:  {r2:.4f}")
    print(f"MAE:       ₹{mae:.2f}")
    print(f"RMSE:      ₹{rmse:.2f}")
    print("-" * 40)

evaluate_model(model_budget, Xb_test, yb_test, "💸 Budget")
evaluate_model(model_mid, Xm_test, ym_test, "🧑‍💻 Mid-Range")
evaluate_model(model_premium, Xp_test, yp_test, "🚀 Premium")
"""
'''
# Step 1: Train-test split first (no leakage)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Step 2: Get thresholds from TRAINING data only
low_thresh = train_df['Price'].quantile(0.33)
high_thresh = train_df['Price'].quantile(0.66)

# Step 3: Segment training data
df_budget_train = train_df[train_df['Price'] <= low_thresh]
df_mid_train = train_df[(train_df['Price'] > low_thresh) & (train_df['Price'] <= high_thresh)]
df_premium_train = train_df[train_df['Price'] > high_thresh]

# Step 4: Segment test data using same thresholds
df_budget_test = test_df[test_df['Price'] <= low_thresh]
df_mid_test = test_df[(test_df['Price'] > low_thresh) & (test_df['Price'] <= high_thresh)]
df_premium_test = test_df[test_df['Price'] > high_thresh]

# Step 5: Log-transform prices
for subset in [df_budget_train, df_mid_train, df_premium_train, df_budget_test, df_mid_test, df_premium_test]:
    subset['Log_Price'] = np.log1p(subset['Price'])

# Step 6: Define features (exclude target columns)
features = [col for col in df.columns if col not in ['Price', 'Log_Price']]

# Step 7: Split into X, y for each segment
# Budget
Xb_train = df_budget_train[features]
yb_train = df_budget_train['Log_Price']
Xb_test = df_budget_test[features]
yb_test = df_budget_test['Log_Price']

# Mid
Xm_train = df_mid_train[features]
ym_train = df_mid_train['Log_Price']
Xm_test = df_mid_test[features]
ym_test = df_mid_test['Log_Price']

# Premium
Xp_train = df_premium_train[features]
yp_train = df_premium_train['Log_Price']
Xp_test = df_premium_test[features]
yp_test = df_premium_test['Log_Price']

# Step 8: Train models for each segment
model_budget = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_budget.fit(Xb_train, yb_train)

model_mid = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_mid.fit(Xm_train, ym_train)

model_premium = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_premium.fit(Xp_train, yp_train)

# Step 9: Evaluation function
def evaluate_model(model, X_test, y_test, segment_name):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    r2 = r2_score(y_test_actual, y_pred)
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

    print(f"📊 {segment_name} Laptops Performance:")
    print(f"R² Score:  {r2:.4f}")
    print(f"MAE:       ₹{mae:.2f}")
    print(f"RMSE:      ₹{rmse:.2f}")
    print("-" * 40)

# Step 10: Evaluate each model
evaluate_model(model_budget, Xb_test, yb_test, "💸 Budget")
evaluate_model(model_mid, Xm_test, ym_test, "🧑‍💻 Mid-Range")
evaluate_model(model_premium, Xp_test, yp_test, "🚀 Premium")
'''


# Step 1: Train-test split first (no leakage)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Step 2: Get thresholds from TRAINING data only
# low_thresh = train_df['Price'].quantile(0.4)


# Step 3: Segment training data
df_budget_train = train_df[train_df['Price'] <40000] #<= low_thresh]
df_mid_premium_train = train_df[train_df['Price'] > 40000] # >low_thresh)]


# Step 4: Segment test data using same thresholds
df_budget_test = test_df[test_df['Price'] <=40000] #<= low_thresh]
df_mid_premium_test = test_df[test_df['Price'] > 40000] #> low_thresh)]


# Step 5: Log-transform prices
for subset in [df_budget_train, df_mid_premium_train, df_budget_test, df_mid_premium_test]:
    subset['Log_Price'] = np.log1p(subset['Price'])

# Step 6: Define features (exclude target columns)
features = [col for col in df.columns if col not in ['Price', 'Log_Price']]

# Step 7: Split into X, y for each segment
# Budget
Xb_train = df_budget_train[features]
yb_train = df_budget_train['Log_Price']
Xb_test = df_budget_test[features]
yb_test = df_budget_test['Log_Price']

# Mid & premium
Xm_train = df_mid_premium_train[features]
ym_train = df_mid_premium_train['Log_Price']
Xm_test = df_mid_premium_test[features]
ym_test = df_mid_premium_test['Log_Price']


# Step 8: Train models for each segment
model_budget = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_budget.fit(Xb_train, yb_train)

model_mid = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model_mid.fit(Xm_train, ym_train)


# Step 9: Evaluation function
def evaluate_model(model, X_test, y_test, segment_name):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    r2 = r2_score(y_test_actual, y_pred)
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

    print(f"📊 {segment_name} Laptops Performance:")
    print(f"R² Score:  {r2:.4f}")
    print(f"MAE:       ₹{mae:.2f}")
    print(f"RMSE:      ₹{rmse:.2f}")
    print("-" * 40)

# Step 10: Evaluate each model
evaluate_model(model_budget, Xb_test, yb_test, "💸 Budget")
evaluate_model(model_mid, Xm_test, ym_test, "🧑‍💻 Mid-Range")



'''
from sklearn.model_selection import RandomizedSearchCV


# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3]
}

# Evaluation function
def evaluate_model(name, model, X_test, y_test):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    r2 = r2_score(y_test_actual, y_pred)
    mae = np.mean(np.abs(y_test_actual - y_pred))
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

    print(f"\n📊 {name} Segment Tuned Performance")
    print(f"R² Score:  {r2:.4f}")
    print(f"MAE:       ₹{mae:.2f}")
    print(f"RMSE:      ₹{rmse:.2f}")
    print("-" * 40)

# Helper function to tune and evaluate
def tune_and_evaluate(name, X_train, y_train, X_test, y_test):
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=30,
        cv=3,
        verbose=1,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error'
    )
    
    tuner.fit(X_train, y_train)
    best_model = tuner.best_estimator_
    
    print(f"\n✅ {name} Best Parameters:")
    print(tuner.best_params_)
    
    evaluate_model(name, best_model, X_test, y_test)
    
    return best_model

# Perform tuning for all three segments
best_model_budget = tune_and_evaluate("💸 Budget", Xb_train, yb_train, Xb_test, yb_test)
best_model_mid    = tune_and_evaluate("🧑‍💻 Mid-Range", Xm_train, ym_train, Xm_test, ym_test)
best_model_premium= tune_and_evaluate("🚀 Premium", Xp_train, yp_train, Xp_test, yp_test)

'''

'''
from sklearn.model_selection import cross_val_score
scores_budget = cross_val_score(model_budget, X_budget, y_budget, cv=5, scoring='r2',n_jobs=-1)
print(scores_budget.mean())

scores_mid = cross_val_score(model_mid, X_mid, y_mid, cv=5, scoring='r2',n_jobs=-1)
print(scores_mid.mean())

scores_premium = cross_val_score(model_premium, X_premium, y_premium, cv=5, scoring='r2',n_jobs=-1)
print(scores_premium.mean())

train_r2 = model_mid.score(X_mid, y_mid)
print("Train R²:", train_r2)
'''


def plot_feature_importance(model, features, segment_name):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-15:]  # Top 15

    plt.figure(figsize=(8, 6))
    plt.title(f"Top Features for {segment_name} Segment")
    plt.barh(range(len(indices)), importance[indices], color="skyblue", align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.show()

# Plot for both segments
plot_feature_importance(model_budget, Xb_train.columns.tolist(), "💸 Budget")
plot_feature_importance(model_mid, Xm_train.columns.tolist(), "🧑‍💻 Mid & Premium")

from sklearn.model_selection import cross_val_score

scores_budget = cross_val_score(
    model_budget, Xb_train, yb_train, cv=5, scoring='r2', n_jobs=-1
)
print("💸 Budget Segment CV R²:", scores_budget)
print("Average R² (Budget):", scores_budget.mean())

scores_mid = cross_val_score(
    model_mid, Xm_train, ym_train, cv=5, scoring='r2', n_jobs=-1
)
print("🧑‍💻 Mid+Premium Segment CV R²:", scores_mid)
print("Average R² (Mid+Premium):", scores_mid.mean())

from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

param_dist = {
    'n_estimators': stats.randint(50, 300),           # Number of trees
    'max_depth': stats.randint(3, 10),                 # Tree depth
    'learning_rate': stats.uniform(0.01, 0.2),         # Step size shrinkage
    'subsample': stats.uniform(0.6, 0.4),              # % of rows per tree
    'colsample_bytree': stats.uniform(0.6, 0.4),       # % of features per tree
    'gamma': stats.uniform(0, 0.5),                    # Minimum loss reduction
    'reg_alpha': stats.uniform(0, 0.5),                # L1 regularization
    'reg_lambda': stats.uniform(0.5, 1.0),             # L2 regularization
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

random_search_budget = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=30,                   # Number of random combinations to try
    cv=5,                        # 5-fold CV
    scoring='r2',                # Can also try 'neg_root_mean_squared_error'
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search_budget.fit(Xb_train, yb_train)

print("✅ Best R² Score (Budget):", random_search_budget.best_score_)
print("🏆 Best Parameters (Budget):", random_search_budget.best_params_)

best_model_budget = random_search_budget.best_estimator_

# Re-evaluate
evaluate_model(best_model_budget, Xb_test, yb_test, "💸 Budget (Tuned)")

random_search_mid = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search_mid.fit(Xm_train, ym_train)

best_model_mid = random_search_mid.best_estimator_
evaluate_model(best_model_mid, Xm_test, ym_test, "🧑‍💻 Mid+Premium (Tuned)")