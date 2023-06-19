import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

from itertools import combinations
import scipy.stats as sts
import seaborn as sns

# for decision tree & random forest & SVM
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# read the data
data = pd.read_csv("../data/thyroidDF.csv")

"""###################################################################################################"""

# get some information on the data
data.describe()
data.dtypes
data.head(5)
data.shape
data.isna().sum(axis=0)
data['target'].value_counts()
data_comp = data.dropna()
data_comp.shape

"""###################################################################################################"""

# Distribution plot before cleaning
numerical_columns = data.select_dtypes(include=['int', 'float']).columns
numerical_columns = [col for col in numerical_columns if col != 'patient_id']
# Create displot for each numerical column
for column in numerical_columns:
    sns.displot(data[column])
    plt.savefig(f"{'../output/displots/displots_init'}/{column}_displot.png")
    plt.clf()

# clean the data
data = data[data['age'] <= 110]
data = data[data['age'] > 0]
data = data.drop_duplicates('patient_id')

# Identify numerical columns based on data types
numerical_cols = data.select_dtypes(include=[np.number]).columns

# Select only numerical columns for imputation
data_numerical = data[numerical_cols]

# Perform most_frequent imputation on numerical data only
imputer = SimpleImputer(strategy='most_frequent')
data_imputed_numerical = pd.DataFrame(imputer.fit_transform(data_numerical), columns=data_numerical.columns)

# Replace original numerical columns with imputed values
data[numerical_cols] = data_imputed_numerical
data = data.dropna()

# Distribution plots after cleaning
numerical_columns = data.select_dtypes(include=['int', 'float']).columns
numerical_columns = [col for col in numerical_columns if col != 'patient_id']
# Create displot for each numerical column
for column in numerical_columns:
    sns.displot(data[column])
    plt.savefig(f"{'../output/displots/displots_cleared'}/{column}_displot_cleaned.png")
    plt.clf()
"""###################################################################################################"""

# some plots to get familiar with the data
data_with_tumor = data[data['tumor'] == 't']
sns.histplot(data=data_with_tumor, x='age', bins=range(0, 100, 10), hue='tumor', multiple='stack')
plt.savefig("../output/data_with_tumor.png")
plt.clf()
# age distribution
sns.histplot(data=data, x="age", color="gray")
plt.savefig("../output/age_distribution.png")
plt.clf()
vars = [col for col in data.columns if data[col].dtype == 'float64' and col != 'patient_id']
pplot = sns.pairplot(data=data,x_vars=vars, y_vars=vars,)
plt.savefig("../output/pair_plot.png")
plt.clf()
"""###################################################################################################"""

# LOGISTIC REGRESSION

# only go for healthy or ill
counts = data['target'].value_counts()
values_keep = counts[counts >= 100].index.tolist()
data = data[data['target'].isin(values_keep)]
data.loc[data['target'] != '-', 'target'] = '+'

# one-hot encoding to make all values numerical
cols_with_letters = data.select_dtypes(include=['object']).columns.tolist()
# Exclude the target column from the list of columns to encode
cols_to_encode = [col for col in cols_with_letters if col != 'target' and col != 'referal_source']
data = pd.get_dummies(data, columns=cols_to_encode)
data.head(5)

# scale continuous data
cols_to_scale = [col for col in data.columns if col != 'patient_id' and data[col].dtype != 'object']
# Scale the selected columns using StandardScaler

numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

# Exclude any columns with only 2 unique values (binary columns)
continuous_cols = [col for col in numeric_cols if len(data[col].unique()) > 2]

# Select the continuous columns from the DataFrame
X = data[continuous_cols]
# Select your target variable
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Fit the logistic regression model
sc = StandardScaler()
tr_scale = sc.fit_transform(X_train)
print(tr_scale.shape)
te_scale = sc.transform(X_test)
lr = LogisticRegression(class_weight='balanced', random_state= 42)
lr.fit(tr_scale, y_train)
# Make predictions on the test set
y_pred = lr.predict(te_scale)
print(te_scale.shape)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

conf_mat = confusion_matrix(y_test, y_pred)
conf_plt = sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('../output/conf_martrix.png')
plt.clf()

# Feature Importance Plot
# Get the absolute coefficient values
coefficients = np.abs(lr.coef_.ravel())
# Get the indices of the top 10 features
top_feature_indices = np.argsort(coefficients)[-10:]
# Get the top 10 feature names
top_feature_names = X_train.columns[top_feature_indices]
# Get the corresponding coefficient values
top_coefficients = coefficients[top_feature_indices]
# Sort the features and coefficients in descending order
top_feature_names = top_feature_names[::-1]
top_coefficients = top_coefficients[::-1]
# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.barh(top_feature_names, top_coefficients)
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Features')
plt.title('Top 10 Feature Importance')
plt.savefig('../output/LR_feature_selection.png')
plt.clf()

coefficients = lr.coef_
importance = np.abs(coefficients)
# Normalize the absolute coefficients to obtain feature importance scores
importance_scores = (importance / np.sum(importance)).flatten()

fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(np.arange(len(importance_scores)), importance_scores)
ax.set_xticks(np.arange(len(importance_scores)), X.columns.tolist())  # set the xticks according to the feature names, and rotate them by 90 degrees
ax.set_title("Normalized feature importance", fontsize=20)
plt.savefig('../output/feature_imp_lr.png')
plt.clf()
"""###################################################################################################"""
"""###################################################################################################"""

# DECISION TREE --> TODO: SCALE?

df = pd.read_csv('../data/thyroidDF.csv')
#df = data

# Create a SimpleImputer object with 'mean' strategy
imputer = SimpleImputer(strategy='mean')
# Impute missing values in the TSH column
df['TSH'] = imputer.fit_transform(df[['TSH']])
# Impute missing values in the T3 column
df['T3'] = imputer.fit_transform(df[['T3']])
# Impute missing values in the TT4 column
df['TT4'] = imputer.fit_transform(df[['TT4']])
# Impute missing values in the T4U column
df['T4U'] = imputer.fit_transform(df[['T4U']])
# Impute missing values in the FTI column
df['FTI'] = imputer.fit_transform(df[['FTI']])
# Impute missing values in the TBG column
df['TBG'] = imputer.fit_transform(df[['TBG']])

# For missing sex values
# Calculate the ratio of males to females
male_count = df[df["sex"] == "M"].shape[0]
female_count = df[df["sex"] == "F"].shape[0]
ratio = male_count / female_count
# Fill in missing sex values with the ratio applied to the missing values
missing_sex_count = df["sex"].isnull().sum()
missing_male_count = int(round(missing_sex_count / (ratio + 1)))
missing_female_count = missing_sex_count - missing_male_count

df.loc[df["sex"].isnull(), "sex"] = ["M"] * missing_male_count + ["F"] * missing_female_count


# One-hot encode 'sex' column
df = pd.get_dummies(df, columns=['sex'])
# Loop through all columns with binary values and one-hot encode them
for col in df.columns:
    if df[col].dtype == 'object' and set(df[col].unique()) == {'t', 'f'}:
        df[col] = df[col].apply(lambda x: 1 if x == 't' else 0)
        df = pd.get_dummies(df, columns=[col])
# Print the updated dataframe
print(df.head())


# Split the dataset into training and testing sets
X = df[['age', 'sex_F', 'sex_M', 'on_thyroxine_1', 'on_thyroxine_0', 'query_hyperthyroid_1',
        'query_hyperthyroid_0', 'query_hypothyroid_1', 'query_hypothyroid_0', 'pregnant_1',
        'pregnant_0', 'thyroid_surgery_1', 'thyroid_surgery_0', 'I131_treatment_1',
        'I131_treatment_0', 'query_on_thyroxine_1', 'query_on_thyroxine_0', 'on_antithyroid_meds_1',
        'on_antithyroid_meds_0', 'sick_1', 'sick_0', 'tumor_1', 'tumor_0', 'lithium_1', 'lithium_0',
        'goitre_1', 'goitre_0', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']]  # Features
y = df['target']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaler
sc = StandardScaler()
X_train_scale = sc.fit_transform(X_train)
X_test_scale = sc.transform(X_test)

# Define and train the decision tree model
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt.fit(X_train_scale, y_train)

# Make predictions on the testing set and evaluate the model performance
y_pred = dt.predict(X_test_scale)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

print('\nClassification Report:\n', classification_report(y_test, y_pred))


"""###################################################################################################"""

# RANDOM FOREST

# Create a list of the feature columns
feature_cols = ['age', 'sex_F', 'sex_M', 'on_thyroxine_1', 'on_thyroxine_0', 'query_hyperthyroid_1',
                'query_hyperthyroid_0', 'query_hypothyroid_1', 'query_hypothyroid_0', 'pregnant_1',
                'pregnant_0', 'thyroid_surgery_1', 'thyroid_surgery_0', 'I131_treatment_1',
                'I131_treatment_0', 'query_on_thyroxine_1', 'query_on_thyroxine_0', 'on_antithyroid_meds_1',
                'on_antithyroid_meds_0', 'sick_1', 'sick_0', 'tumor_1', 'tumor_0', 'lithium_1', 'lithium_0',
                'goitre_1', 'goitre_0', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']

# Create a dataframe of the feature data
X = df[feature_cols]

# Create a series of the target variable
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaler
sc = StandardScaler()
X_train_scale = sc.fit_transform(X_train)
X_test_scale = sc.transform(X_test)

# Create a random forest classifier object with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Fit the model to the training data
rf.fit(X_train_scale, y_train)

# Use the model to make predictions on the test data
y_pred = rf.predict(X_test_scale)

# Calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.6f}')
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Feature Importance Plot
feature_importances = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=['importance'])
feature_importances.sort_values(by='importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y=feature_importances.index, data=feature_importances)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.savefig('../output/RF_feature_importance.png')
plt.clf()

# Confusion Matrix Plot
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('../output/RF_conf_matrix.png')
plt.clf()

"""###################################################################################################"""

#SVM
# extract the predictor variables and the target variable
X = df.drop(['target', 'patient_id'], axis=1)
y = df['target']

# encode categorical variables
# get a list of columns where the data type is object
object_cols = list(X.select_dtypes(include=['object']).columns)
# perform one-hot encoding on categorical variables
X = pd.get_dummies(X, columns=object_cols)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#scale the sets

sc = StandardScaler()
X_train_scale = sc.fit_transform(X_train)
X_test_scale = sc.transform(X_test)

# create the SVM model
model = SVC()

# fit the model to the training data
model.fit(X_train_scale, y_train)

# make predictions on the test data
y_pred = model.predict(X_test_scale)

# evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Performance of the logistic regression model:")
print("Accuracy: {:.3f}; Precision: {:.3f}; Recall: {:.3f}; F1 score: {:.3f}".format(accuracy, precision, recall, f1))
