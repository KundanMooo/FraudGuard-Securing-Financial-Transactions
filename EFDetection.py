# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

# Create a copy of the original dataset
dforg = df.copy()

# Display the 'type' column from the original dataset
dforg['type']

# Data Overview

# Display the first few rows of the dataset
df.head()

# Get the shape of the dataset (number of rows and columns)
df.shape

# Generate summary statistics for the dataset
df.describe()

# Get information about the dataset, including data types and missing values
df.info()

# Display the number of unique values for each column
df.nunique()

# Check for missing values in the dataset
df.isna().sum().sum()

# Check for duplicate rows in the dataset
df.duplicated().sum()

# Data Preprocessing

# Separate categorical and numerical columns
categorical_columns = []
numerical_columns = []

for column in df.columns:
    if df[column].dtype == 'object':
        categorical_columns.append(column)
    else:
        numerical_columns.append(column)

# Extend the categorical columns list with 'isFraud' and 'isFlaggedFraud'
categorical_columns.extend(['isFraud', 'isFlaggedFraud'])

# Display the categorical columns
df[categorical_columns]

# Remove 'isFraud' and 'isFlaggedFraud' from the numerical columns
numerical_columns.remove('isFraud')
numerical_columns.remove('isFlaggedFraud')

# Display the numerical columns
df[numerical_columns]

# Data Visualization

# Plot a histogram of the distribution of hourly fraud in electronic transactions
plt.figure(figsize=(10, 6))
df['step'].value_counts().plot(kind='hist', bins=100)
plt.xticks(range(0, 50000, 1000), rotation=90)
plt.suptitle("Distribution of Hourly Fraud in Electronic Transactions")
plt.title("Most Fraud Occurs Within 1000 Hours")
plt.show()

# Box Plot

# Create a box plot for each numerical column after removing outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_columns])
plt.title("Box Plots of Numerical Columns")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram

# Create histograms for each numerical column
plt.figure(figsize=(16, 10))
for i, col in enumerate(numerical_columns, start=1):
    plt.subplot(2, 4, i)
    plt.hist(df[col], bins=30)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

# Share of Online Transaction Types in Fraud
df['type'].value_counts().plot(kind='bar')

# Explore the relationship between old and new balances before and after suspicious transactions
df1 = df[df['isFraud'] == 1]
result_all = df.groupby('type')['amount'].sum()
result_fraud = df1.groupby('type')['amount'].sum()

# Combine both results into a single DataFrame
combined_result = pd.DataFrame({'All Transactions': result_all, 'Fraud Transactions': result_fraud})

# Plot a stacked bar chart
ax = combined_result.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#1DA1F2', '#DF1B12'])
plt.xlabel('Transaction Type')
plt.ylabel('Total Amount')
plt.title('Electronic Fraud Occurrence by Transaction Type')
plt.tight_layout()
plt.show()

# Create a scatter plot to visualize balance changes before and after fraudulent transactions
plt.figure(figsize=(10, 6))
plt.scatter(df['oldbalanceOrg'], df['newbalanceOrig'], c=df['isFraud'], cmap='coolwarm', alpha=0.5)
plt.xlabel('Old Balance')
plt.ylabel('New Balance')
plt.title('Balance Change Before and After Fraudulent Transactions')
plt.colorbar(label='Is Fraud')
plt.tight_layout()
plt.show()

# Box Plot of Balance Changes Before Fraud
plt.figure(figsize=(8, 6))
sns.boxplot(x='isFraud', y='oldbalanceOrg', data=df)
plt.title('Box Plot of Balance Changes Before Fraud')
plt.xlabel('Is Fraud')
plt.ylabel('Balance Change')
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.show()

# Violin Plot of Balance Changes Before Fraud
plt.figure(figsize=(8, 6))
sns.violinplot(x='isFraud', y='oldbalanceOrg', data=df)
plt.title('Violin Plot of Balance Changes Before Fraud')
plt.xlabel('Is Fraud')
plt.ylabel('Balance Change')
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.show()

# Box Plot showing Average Balance Before and After Fraud
plt.figure(figsize=(8, 6))
plt.bar(['Before Fraud', 'After Fraud'], [average_original_balance, average_new_balance], yerr=[std_original_balance, std_new_balance], capsize=10)
plt.title('Average Balance Before and After Fraud')
plt.ylabel('Average Balance')
plt.show()

# Box Plot of all numerical columns
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_columns])
plt.title("Box Plots of Numerical Columns")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature Engineering

# Create a copy of the dataset for feature engineering
dfn = df.copy()

# Drop unnecessary columns
dfn = dfn.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# Create new features
dfn['step_day_week'] = dfn['step'] % 7
dfn['step_month'] = (dfn['step'] - 1) // 30 + 1
dfn['log_amount'] = np.log1p(dfn['amount'])
dfn['sqr_amount'] = np.sqrt(dfn['amount'])
dfn['diff_org'] = dfn['newbalanceOrig'] - dfn['oldbalanceOrg']
dfn['diff_Dest'] = dfn['newbalanceDest'] - dfn['oldbalanceDest']
dfn['amountmeanrolling3'] = dfn['amount'].rolling(window=3).mean()
dfn['amountsumrolling7'] = dfn['amount'].rolling(window=7).sum()
dfn['amount+oldorg'] = dfn['amount'] * dfn['oldbalanceOrg']
dfn['amount+neworg'] = dfn['amount'] * dfn['newbalanceOrig']

# Perform one-hot encoding on the 'type' column
dfn_e = pd.get_dummies(dfn['type'], prefix='type', drop_first=True)
dfn = pd.concat([dfn_e, dfn], axis=1)

# Remove rows with missing values
dfn = dfn.dropna()

# Select relevant features based on correlation
list_imp_feature = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                    'diff_org', 'diff_Dest', 'log_amount', 'sqr_amount', 'amountmeanrolling3', 'amountsumrolling7',
                    'amount+oldorg', 'amount+neworg', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

dfnn = dfn[list_imp_feature]

# Handle multicollinearity by removing highly correlated features
correlation_threshold = 0.7  # Set the correlation threshold
correlation_matrix = dfnn.corr().abs()  # Calculate the correlation matrix
mask = correlation_matrix >= correlation_threshold  # Create a mask for features to drop
features_to_drop = set()  # Get a set of feature names to drop

for i in range(len(dfnn.columns)):
    for j in range(i + 1, len(dfnn.columns)):
        if mask.iloc[i, j]:
            colname_i = dfnn.columns[i]
            colname_j = dfnn.columns[j]
            if colname_i not in features_to_drop:
                features_to_drop.add(colname_j)

# Drop highly correlated features from the DataFrame
dfnn = dfnn.drop(columns=features_to_drop)

# Normalize the dataset using Min-Max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(dfnn[list_imp_feature])
scaled_df = dfnn.copy()
scaled_df[list_imp_feature] = scaled_features

# Resampling to address class imbalance using SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(scaled_df.drop('isFraud', axis=1), scaled_df['isFraud'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_logistic = logistic_regression.predict(X_test)

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)

# XGBoost Classifier
xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)

# Random Forest Classifier
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)

# Evaluate the models
logistic_regression_accuracy = accuracy_score(y_test, y_pred_logistic)
decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
random_forest_accuracy = accuracy_score(y_test, y_pred_rf)

# Print model evaluation metrics
print("Logistic Regression:")
print("Accuracy:", logistic_regression_accuracy)
print()

print("Decision Tree Classifier:")
print("Accuracy:", decision_tree_accuracy)
print()

print("XGBoost Classifier:")
print("Accuracy:", xgb_accuracy)
print()

print("Random Forest Classifier:")
print("Accuracy:", random_forest_accuracy)
print()
