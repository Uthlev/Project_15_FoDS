import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV


# read the data
data = pd.read_csv("../data/thyroidDF.csv")

"""###################################################################################################"""

# get some information on the data
print("raw data description:\n" , data.describe())
data.dtypes
data.head(5)
print("raw data shape:" , data.shape)
print("raw data NaN:\n", data.isna().sum(axis=0))
print("raw data target:\n",data['target'].value_counts())
data_comp = data.dropna()
print("raw data dropna:", data_comp.shape)
print("raw data duplicates", data.duplicated().sum())

"""###################################################################################################"""

# Distribution plot before cleaning
numerical_columns = data.select_dtypes(include=['int', 'float']).columns
numerical_columns = [col for col in numerical_columns if col != 'patient_id']
num_plots = len(numerical_columns)
num_rows = (num_plots + 1) // 2  # Calculate the number of rows
fig, axs = plt.subplots(4, 2, figsize=(10,12))
for i, column in enumerate(numerical_columns):
    row = i // 2
    col = i % 2
    sns.histplot(data[column], ax=axs[row, col], bins=100)
    axs[row, col].set_title(f'{column} distribution plot')
if num_plots % 2 == 1:
    fig.delaxes(axs[-1, -1])
plt.tight_layout()
plt.savefig('../output/displots/displots_init/displots_init.png', dpi = 600)
plt.clf()

thresholds = {
    'age': 110,
    'TSH': 20,
    'T3': 10,
    'TT4': 320,
    'T4U': 1000,
    'FTI': 400,
    'TBG': 50
}
# Iterate over columns and thresholds
for column, threshold in thresholds.items():
    # Calculate the most frequent value in the column
    most_frequent_value = data[column].mode()[0]
    # Create a boolean mask to identify values above the threshold
    mask = data[column] > threshold
    # Replace values above the threshold with the most frequent value
    data.loc[mask, column] = most_frequent_value

data = data.drop_duplicates('patient_id')

# Select only numerical columns for imputation
data_numerical = data[numerical_columns]

# Perform most_frequent imputation on numerical data only
imputer = SimpleImputer(strategy='most_frequent')
data_imputed_numerical = pd.DataFrame(imputer.fit_transform(data_numerical), columns=data_numerical.columns)

# Replace original numerical columns with imputed values
data[numerical_columns] = data_imputed_numerical

# For missing sex values
# Calculate the ratio of males to females
male_count = data[data["sex"] == "M"].shape[0]
female_count = data[data["sex"] == "F"].shape[0]
ratio = male_count / female_count
# Fill in missing sex values with the ratio applied to the missing values
missing_sex_count = data["sex"].isnull().sum()
missing_male_count = int(round(missing_sex_count / (ratio + 1)))
missing_female_count = missing_sex_count - missing_male_count

data.loc[data["sex"].isnull(), "sex"] = ["M"] * missing_male_count + ["F"] * missing_female_count

data.loc[(data['target'] == 'A') | (data['target'] == 'B') | (data['target'] == 'C') | (data['target'] == 'AK') | (data['target'] == 'D'), 'target'] = '1 - hyperthyroid conditions'
data.loc[(data['target'] == 'E') | (data['target'] == 'F') | (data['target'] == 'FK') | (data['target'] == 'G') | (data['target'] == 'GI') | (data['target'] == 'GK') | (data['target'] == 'GKJ'), 'target'] = '2 - hypothyroid conditions'
data.loc[(data['target'] == 'I') | (data['target'] == 'J') | (data['target'] == 'C|I'), 'target'] = '3 - binding protein'
data.loc[(data['target'] == 'K') | (data['target'] == 'KJ') | (data['target'] == 'H|K'), 'target'] = '4 - general health'
data.loc[(data['target'] == 'L') | (data['target'] == 'M') | (data['target'] == 'MK') | (data['target'] == 'N') | (data['target'] == 'MI') | (data['target'] == 'LJ'), 'target'] = '5 - replacement therapy'
data.loc[(data['target'] == 'O') | (data['target'] == 'P') | (data['target'] == 'Q') | (data['target'] == 'OI'), 'target'] = '6 - antithyroid treatment'
data.loc[(data['target'] == 'R') | (data['target'] == 'S') | (data['target'] == 'T') | (data['target'] == 'D|R'), 'target'] = '7 - miscellaneous'


# Distribution plots after cleaning
numerical_columns = data.select_dtypes(include=['int', 'float']).columns
numerical_columns = [col for col in numerical_columns if col != 'patient_id']
num_plots = len(numerical_columns)
num_rows = (num_plots + 1) // 2  # Calculate the number of rows
fig, axs = plt.subplots(4, 2, figsize=(10,12))
for i, column in enumerate(numerical_columns):
    row = i // 2
    col = i % 2
    sns.histplot(data[column], ax=axs[row, col], bins=100)
    axs[row, col].set_title(f'{column} distribution plot')
if num_plots % 2 == 1:
    fig.delaxes(axs[-1, -1])
plt.tight_layout()
plt.savefig('../output/displots/displots_cleared/displots_cleaned.png', dpi = 600)
plt.clf()

# get some information on the data after cleaning
print("cleared data description:\n" , data.describe())
data.dtypes
data.head(5)
print("cleared data shape:" , data.shape)
print("cleared data NaN\n:", data.isna().sum(axis=0))
print("cleared data target\n:",data['target'].value_counts())
data_comp = data.dropna()
print("cleared data dropna:", data_comp.shape)

"""###################################################################################################"""

# some plots to get familiar with the data
data_with_tumor = data[data['tumor'] == 't']
sns.histplot(data=data_with_tumor, x='age', bins=range(0, 100, 10), hue='tumor', multiple='stack')
plt.tight_layout()
plt.savefig("../output/data_with_tumor.png")
plt.clf()

# age distribution
sns.histplot(data=data, x="age", color="gray")
plt.tight_layout()
plt.savefig("../output/age_distribution.png")
plt.clf()

#distribution plots
binary_columns = data.select_dtypes(include=['object']).columns
bin_plots = len(binary_columns)
bin_rows = (bin_plots + 3) // 4  # Calculate the number of rows for 4 columns

fig, axs = plt.subplots(bin_rows, 4, figsize=(16, 12))  # Increase figsize and set 4 columns

for i, column in enumerate(binary_columns):
    row = i // 4  # Adjust for 4 columns
    col = i % 4  # Adjust for 4 columns
    sns.histplot(data[column], ax=axs[row, col])
    axs[row, col].set_title(f'{column} distribution plot')

if bin_plots % 4 != 0:
    # Remove empty subplots
    for i in range(bin_plots % 4, 4):
        fig.delaxes(axs[-1, i])

plt.tight_layout()
plt.savefig('../output/displots/displots_binstr/binary_displots.png', dpi=600)
plt.clf()

vars = [col for col in data.columns if data[col].dtype == 'float64' and col != 'patient_id']
pplot = sns.pairplot(data=data,x_vars=vars, y_vars=vars,)
plt.tight_layout()
plt.savefig("../output/pair_plot.png")
plt.clf()
"""###################################################################################################"""

# DECISION TREE

# One-hot encode 'sex' column
data = pd.get_dummies(data, columns=['sex'])
# Loop through all columns with binary values and one-hot encode them
for col in data.columns:
    if data[col].dtype == 'object' and set(data[col].unique()) == {'t', 'f'}:
        data[col] = data[col].apply(lambda x: 1 if x == 't' else 0)
        data = pd.get_dummies(data, columns=[col])
# Print the updated dataframe
print("data head:", data.head())

#df = pd.read_csv('../data/thyroidDF.csv')
df = data

# Split the dataset into training and testing sets
X = df[['age', 'sex_F', 'sex_M', 'on_thyroxine_1', 'on_thyroxine_0', 'query_hyperthyroid_1',
        'query_hyperthyroid_0', 'query_hypothyroid_1', 'query_hypothyroid_0', 'pregnant_1',
        'pregnant_0', 'thyroid_surgery_1', 'thyroid_surgery_0', 'I131_treatment_1',
        'I131_treatment_0', 'query_on_thyroxine_1', 'query_on_thyroxine_0', 'on_antithyroid_meds_1',
        'on_antithyroid_meds_0', 'sick_1', 'sick_0', 'tumor_1', 'tumor_0', 'lithium_1', 'lithium_0',
        'goitre_1', 'goitre_0', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]  # Features --> TBG nicht!

#X = df.drop(['target', 'patient_id', 'referral_source'], axis=1)
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
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Performance of the Decision Tree model:")
print("Accuracy: {:.3f} \nPrecision: {:.3f} \nRecall: {:.3f} \nF1 score: {:.3f}".format(accuracy, precision, recall, f1))

print('\nDecision Tree Classification Report:\n', classification_report(y_test, y_pred))

# Plots
# Feature Importance Plot
feature_importances = pd.DataFrame(dt.feature_importances_, index=X_train.columns, columns=['importance'])
feature_importances.sort_values(by='importance', ascending=True, inplace=True)
top_10_features = feature_importances.tail(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=top_10_features.index, data=top_10_features, color='tab:blue')
plt.title('Top 10 Decision Tree Feature Importances', fontsize=18)
plt.xlabel('Importance', fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('../output/DT_feature_importance.png', dpi = 600)
plt.clf()

# confusion matrix
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.4)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('DT: Confusion Matrix', fontsize=20)
plt.tight_layout()
plt.savefig('../output/DT_conf_matrix.png', dpi=600)
plt.clf()
sns.reset_defaults()

# Tree Plot with max_depth = 3
dt = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=4)
dt.fit(X_train_scale, y_train)
plt.figure(figsize=(37, 8))
class_labels = np.unique(y)
plot_tree(dt, filled=True, rounded=True, feature_names=X.columns, class_names=class_labels, fontsize=10)
plt.tight_layout()
plt.savefig('../output/Tree Plot', dpi=600)
plt.clf()

"""###################################################################################################"""

# RANDOM FOREST

# Create a list of the feature columns
feature_cols = ['age', 'sex_F', 'sex_M', 'on_thyroxine_1', 'on_thyroxine_0', 'query_hyperthyroid_1',
                'query_hyperthyroid_0', 'query_hypothyroid_1', 'query_hypothyroid_0', 'pregnant_1',
                'pregnant_0', 'thyroid_surgery_1', 'thyroid_surgery_0', 'I131_treatment_1',
                'I131_treatment_0', 'query_on_thyroxine_1', 'query_on_thyroxine_0', 'on_antithyroid_meds_1',
                'on_antithyroid_meds_0', 'sick_1', 'sick_0', 'tumor_1', 'tumor_0', 'lithium_1', 'lithium_0',
                'goitre_1', 'goitre_0', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'] # ohne TBG!

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
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Performance of the Random Forest model:")
print("Accuracy: {:.3f} \nPrecision: {:.3f} \nRecall: {:.3f} \nF1 score: {:.3f}".format(accuracy, precision, recall, f1))

print('\nRandom Forest Classification Report:\n', classification_report(y_test, y_pred))

# Feature Importance Plot
feature_importances = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=['importance'])
feature_importances.sort_values(by='importance', ascending=True, inplace=True)
top_10_features = feature_importances.tail(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=top_10_features.index, data=top_10_features, color='tab:blue')
plt.title('Top 10 Random Forest Feature Importances', fontsize=18)
plt.xlabel('Importance', fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('../output/RF_feature_importance.png', dpi=600)
plt.clf()

# Confusion Matrix Plot
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('RF: Confusion Matrix', fontsize=20)
plt.tight_layout()
plt.savefig('../output/RF_conf_matrix.png', dpi=600)
plt.clf()
sns.reset_defaults()

"""###################################################################################################"""

#SVM for Multiclass
# extract the predictor variables and the target variable
X = df.drop(['target', 'patient_id', 'TBG'], axis=1)
y = df['target']

# encode categorical variables
# get a list of columns where the data type is object
object_cols = list(X.select_dtypes(include=['object']).columns)
# perform one-hot encoding on categorical variables
X = pd.get_dummies(X, columns=object_cols)

##### feature selection #####

# Define the range of k values
k_values = range(10, 15)  #### adjust here computational cost ##### ###best value is around 12
best_f1_score = 0.0
best_result = None

for k in k_values:
    print("trial:", k)
    # Apply SelectKBest feature selection
    k_best = SelectKBest(score_func=f_classif, k=k)  # Choose the desired number of features (k)
    X_selected = k_best.fit_transform(X, y)

    # Get the indices of the selected features
    selected_indices = k_best.get_support(indices=True)

    # Get the names of the selected features (assuming X is a DataFrame)
    selected_features = X.columns[selected_indices]

    # Print the selected feature names
    # print("Selected features for k =", k)
    # print(selected_features)

    ##### hyperparameter tuning ######

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train_scale = sc.fit_transform(X_train)
    X_test_scale = sc.transform(X_test)

    # Define the parameter grid
    param_grid = {
        'C': randint(1, 100),  # List of discrete values for C
        'gamma': [0.001, 0.01, 0.1],  # List of discrete values for gamma
        'kernel': ['linear', 'rbf', 'poly']  # List of kernel options
    }

    # Create an SVM classifier
    svm = SVC()

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=svm, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)

    # Perform random search to find the best hyperparameters
    random_search.fit(X_train_scale, y_train)

    # Evaluate the best model on the testing set
    best_model = random_search.best_estimator_
    accuracy = best_model.score(X_test_scale, y_test)

    # Print the best hyperparameters and the evaluation score
    # print("Best Hyperparameters for k =", k)
    # print(random_search.best_params_)

    # Perform cross-validation
    cv_scores = cross_val_score(best_model, X_train_scale, y_train, cv=5)

    # Fit the model to the training data
    best_model.fit(X_train_scale, y_train)

    # Make predictions on the test data
    y_pred = best_model.predict(X_test_scale)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # print("Cross-Validation Scores:", cv_scores)
    # print("Average Cross-Validation Score:", np.mean(cv_scores))
    # print("Performance of the SVM model:")
    # print("Accuracy: {:.3f}; Precision: {:.3f}; Recall: {:.3f}; F1 score: {:.3f}".format(accuracy, precision, recall, f1))

    # Update the best k and the corresponding results if the F1 score is better
    if f1 > best_f1_score:
        best_f1_score = f1
        best_result = {
            'k': k,
            'selected_features': selected_features,
            'best_model': best_model,
            'X_test_scale': X_test_scale,
            'y_test': y_test,
            'best_hyperparameters': random_search.best_params_,
            'best_accuracy': accuracy,
            'cv_scores': cv_scores,
            'average_cv_score': np.mean(cv_scores),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

# print(best_result)

print("Best k:", best_result['k'])
print("Best Hyperparameters:", best_result['best_hyperparameters'])
print("Best Feature selection:", best_result['selected_features'])
print("Cross-Validation Scores:", best_result['cv_scores'])
print("Average Cross-Validation Score:", best_result['average_cv_score'])
print("Performance of the SVM model:")
print("Accuracy: {:.3f}; Precision: {:.3f}; Recall: {:.3f}; F1 score: {:.3f}".format(best_result['accuracy'], best_result['precision'], best_result['recall'], best_result['f1_score']))


#### feature importance

# Compute permutation importances
perm_importance = permutation_importance(best_result['best_model'], best_result['X_test_scale'], best_result['y_test'], n_repeats=10, random_state=42)
feature_importance = perm_importance.importances_mean

# Create a DataFrame to store the importances and feature names
feature_df = pd.DataFrame({'Feature': best_result['selected_features'], 'Importance': feature_importance})

# Sort the DataFrame by importance in descending order
feature_df = feature_df.sort_values('Importance', ascending=False)

# Get the top 10 features with highest importances
top_features = feature_df.head(10)

# Get the bottom 10 features with lowest importances
#bottom_features = feature_df.tail(10)

# Print the top 10 features
print("Top 10 features:")
print(top_features)

# Print the bottom 10 features
#print("Bottom 10 features:")
#print(bottom_features)

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Multiclass SVM: Top 10 Features')
plt.tight_layout()
#plt.show()
plt.savefig('../output/SVM_feature_selection_multiclass.png')
plt.clf()


'''###################################################################################################'''

# LOGISTIC REGRESSION

# only go for thyroid disease or no thyroid disease
data.loc[(data['target'] == '1 - hyperthyroid conditions') | (data['target'] == '2 - hypothyroid conditions'), 'target'] = '+'
data.loc[(data['target'] == '3 - binding protein') | (data['target'] == '4 - general health') |
         (data['target'] == '5 - replacement therapy') | (data['target'] == '6 - antithyroid treatment') |
         (data['target'] == '7 - miscellaneous'), 'target'] = '-'
dava = data['target'].value_counts()
print(dava)

### one-hot encoding ###

# one-hot encoding to make all values numerical
cols_with_letters = data.select_dtypes(include=['object']).columns.tolist()
# Exclude the target column from the list of columns to encode
cols_to_encode = [col for col in cols_with_letters if col != 'target' and col != 'referal_source']

#activate pd.get_dummies for one hot encoding!
#data = pd.get_dummies(data, columns=cols_to_encode)

# choose the columns to scale?
cols_to_scale = [col for col in data.columns if col != 'patient_id' and data[col].dtype != 'object' and col != 'TBG']

### only using numerica features ###
# Exclude any columns with only 2 unique values (binary columns)
continuous_cols = [col for col in cols_to_scale if len(data[col].unique()) > 2]

### Binary encoding for binary objects ###
# ! deactivate this part when doing the one-hot encoding approach as 'referal_source' wouldn't be found !
# Identify binary object features
binary_cols = [col for col in cols_with_letters if col != 'target' and col != 'referal_source'
               and len(data[col].unique()) == 2]
# Apply binary encoding for binary object features
for col in binary_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
selected_cols = binary_cols + cols_to_scale

# Select the continuous columns from the DataFrame
#1. Only using the numerical features:  choose X=data[continuous_cols]
#2. One-hot encoding:                   choose X=data[cols_to_scale] & activate pd.get_dummies
#3. Binary encoding for binary objects: choose X=data[selected_cols]
# Beware that the same adjustments need to be made for 5-fold CV below!
X = data[continuous_cols]
# Select your target variable
y = data['target']

# Perform feature selection using correlation-based method
selector = SelectKBest(score_func=f_classif, k=6)  # Select the top k features
X_selected = selector.fit_transform(X, y)

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
print('##############################################################')
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
fig, ax = plt.subplots()
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat = conf_mat[[1, 0], :][:, [1, 0]]
conf_plt = sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax = ax)
conf_plt.set(title = 'Confusion Matrix Logistic Regression',
             ylabel = 'True label',
             xlabel = 'Predicted label')

plt.tight_layout()
plt.savefig('../output/conf_martrix.png', dpi = 600)
plt.clf()

### Feature Importance Plot ###

# Fit the SelectKBest feature selector
#selector = SelectKBest(score_func=f_classif, k=6)
selector.fit(X, y)

# Get the scores and corresponding feature names
feature_scores = selector.scores_
feature_names = X.columns

# Sort the feature scores and names in descending order
sorted_indices = np.argsort(feature_scores)[::-1]
sorted_scores = feature_scores[sorted_indices]
sorted_names = feature_names[sorted_indices]

# Select only the top 6 features
top_scores = sorted_scores[:6]
top_names = sorted_names[:6]

# Plot the feature selection
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_scores)), top_scores, align='center')
plt.yticks(range(len(top_scores)), top_names)
plt.xlabel('Coefficient Magnitude', fontsize = 15)
plt.ylabel('Features', fontsize = 15)
plt.title('Top 6 Feature Selection', fontsize = 15)
plt.tight_layout()
plt.savefig('../output/LR_feature_selection.png', dpi = 600)
plt.clf()


### Perform 5-fold cross-validation ###

# Select the continuous columns from the DataFrame
#1. Only using the numerical features:  choose X=data[continuous_cols]
#2. One-hot encoding:                   choose X=data[cols_to_scale] & activate pd.get_dummies
#3. Binary encoding for binary objects: choose X=data[selected_cols]
X = data[continuous_cols]
# Select your target variable
y = data['target']

# Standardize the features
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Create the logistic regression model
lr = LogisticRegression(class_weight='balanced', random_state=42)

# Perform 5-fold cross-validation and get the predictions
y_pred = cross_val_predict(lr, X_scaled, y, cv=5)
scores = cross_val_score(lr, X_scaled, y, cv=5)

# Generate the classification report
report = classification_report(y, y_pred)

# Print the classification report
print("5-fold Classification Report:\n", report)
print("5-fold Cross-validation scores:", scores)
print("5-fold Mean cross-validation score:", scores.mean())
print('')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit the logistic regression model on the training data
lr.fit(X_train, y_train)

# Get the predictions on the test set
y_pred_test = lr.predict(X_test)
# Calculate and print the evaluation metrics on the test set
print("5-fold Accuracy:", accuracy_score(y_test, y_pred_test))
print("5-fold Precision:", precision_score(y_test, y_pred_test, average='weighted', zero_division=0))
print("5-fold Recall:", recall_score(y_test, y_pred_test, average='weighted'))
print("5-fold F1-score:", f1_score(y_test, y_pred_test, average='weighted'))
print('##############################################################')

'''###################################################################################################'''

"####SVM#####################################"

#SVM for binary

df = data

# extract the predictor variables and the target variable
X = df.drop(['target', 'patient_id', 'TBG'], axis=1)
y = df['target']

# encode categorical variables
# get a list of columns where the data type is object
object_cols = list(X.select_dtypes(include=['object']).columns)
# perform one-hot encoding on categorical variables
X = pd.get_dummies(X, columns=object_cols)

##### feature selection #####

# Define the range of k values
k_values = range(12, 16)  #### adjust here computational cost ##### ###best value is 14
best_f1_score = 0.0
best_result = None

for k in k_values:
    print("trial:", k)
    # Apply SelectKBest feature selection
    k_best = SelectKBest(score_func=f_classif, k=k)  # Choose the desired number of features (k)
    X_selected = k_best.fit_transform(X, y)

    # Get the indices of the selected features
    selected_indices = k_best.get_support(indices=True)

    # Get the names of the selected features (assuming X is a DataFrame)
    selected_features = X.columns[selected_indices]

    # Print the selected feature names
    # print("Selected features for k =", k)
    # print(selected_features)

    ##### hyperparameter tuning ######

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train_scale = sc.fit_transform(X_train)
    X_test_scale = sc.transform(X_test)

    # Define the parameter grid
    param_grid = {
        'C': randint(1, 100),  # List of discrete values for C
        'gamma': [0.001, 0.01, 0.1],  # List of discrete values for gamma
        'kernel': ['linear', 'rbf', 'poly']  # List of kernel options
    }

    # Create an SVM classifier
    svm = SVC()

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=svm, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)

    # Perform random search to find the best hyperparameters
    random_search.fit(X_train_scale, y_train)

    # Evaluate the best model on the testing set
    best_model = random_search.best_estimator_
    accuracy = best_model.score(X_test_scale, y_test)

    # Print the best hyperparameters and the evaluation score
    # print("Best Hyperparameters for k =", k)
    # print(random_search.best_params_)

    # Perform cross-validation
    cv_scores = cross_val_score(best_model, X_train_scale, y_train, cv=5)

    # Fit the model to the training data
    best_model.fit(X_train_scale, y_train)

    # Make predictions on the test data
    y_pred = best_model.predict(X_test_scale)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # print("Cross-Validation Scores:", cv_scores)
    # print("Average Cross-Validation Score:", np.mean(cv_scores))
    # print("Performance of the SVM model:")
    # print("Accuracy: {:.3f}; Precision: {:.3f}; Recall: {:.3f}; F1 score: {:.3f}".format(accuracy, precision, recall, f1))

    # Update the best k and the corresponding results if the F1 score is better
    if f1 > best_f1_score:
        best_f1_score = f1
        best_result = {
            'k': k,
            'selected_features': selected_features,
            'best_model': best_model,
            'X_test_scale': X_test_scale,
            'y_test': y_test,
            'best_hyperparameters': random_search.best_params_,
            'best_accuracy': accuracy,
            'cv_scores': cv_scores,
            'average_cv_score': np.mean(cv_scores),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

# print(best_result)

print("Best k:", best_result['k'])
print("Best Hyperparameters:", best_result['best_hyperparameters'])
print("Best Feature selection:", best_result['selected_features'])
print("Cross-Validation Scores:", best_result['cv_scores'])
print("Average Cross-Validation Score:", best_result['average_cv_score'])
print("Performance of the SVM model:")
print("Accuracy: {:.3f}; Precision: {:.3f}; Recall: {:.3f}; F1 score: {:.3f}".format(best_result['accuracy'], best_result['precision'], best_result['recall'], best_result['f1_score']))


#### feature importance

# Compute permutation importances
perm_importance = permutation_importance(best_result['best_model'], best_result['X_test_scale'], best_result['y_test'], n_repeats=10, random_state=42)
feature_importance = perm_importance.importances_mean

# Create a DataFrame to store the importances and feature names
feature_df = pd.DataFrame({'Feature': best_result['selected_features'], 'Importance': feature_importance})

# Sort the DataFrame by importance in descending order
feature_df = feature_df.sort_values('Importance', ascending=False)

# Get the top 10 features with highest importances
top_features = feature_df.head(10)

# Get the bottom 10 features with lowest importances
#bottom_features = feature_df.tail(10)

# Print the top 10 features
print("Top 10 features:")
print(top_features)

# Print the bottom 10 features
#print("Bottom 10 features:")
#print(bottom_features)

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Binary SVM: Top 10 Features')
plt.tight_layout()
#plt.show()
plt.savefig('../output/SVM_feature_selection_binary.png', dpi = 600)
plt.clf()


'''###################################################################################################'''

#### Binary Decision Tree ###

df = data

# Split the dataset into training and testing sets
X = df[['age', 'sex_F', 'sex_M', 'on_thyroxine_1', 'on_thyroxine_0', 'query_hyperthyroid_1',
        'query_hyperthyroid_0', 'query_hypothyroid_1', 'query_hypothyroid_0', 'pregnant_1',
        'pregnant_0', 'thyroid_surgery_1', 'thyroid_surgery_0', 'I131_treatment_1',
        'I131_treatment_0', 'query_on_thyroxine_1', 'query_on_thyroxine_0', 'on_antithyroid_meds_1',
        'on_antithyroid_meds_0', 'sick_1', 'sick_0', 'tumor_1', 'tumor_0', 'lithium_1', 'lithium_0',
        'goitre_1', 'goitre_0', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]  # Features --> TBG nicht!

#X = df.drop(['target', 'patient_id', 'referral_source'], axis=1)
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
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Performance of the Binary Decision Tree model:")
print("Accuracy: {:.3f} \nPrecision: {:.3f} \nRecall: {:.3f} \nF1 score: {:.3f}".format(accuracy, precision, recall, f1))

print('\nBinary Decision Tree Classification Report:\n', classification_report(y_test, y_pred))

# Plots
# Feature Importance Plot
feature_importances = pd.DataFrame(dt.feature_importances_, index=X_train.columns, columns=['importance'])
feature_importances.sort_values(by='importance', ascending=True, inplace=True)
top_10_features = feature_importances.tail(10)

#feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=top_10_features.index, data=top_10_features, color='tab:blue')
plt.title('Top 10 Binary Decision Tree Feature Importances', fontsize=18)
plt.xlabel('Importance', fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('../output/Bin_DT_feature_importance.png', dpi = 600)
plt.clf()

#confusion matrix
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.4)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('Binary DT: Confusion Matrix', fontsize=20)
plt.tight_layout()
plt.savefig('../output/Bin_DT_conf_matrix.png', dpi=600)
plt.clf()
sns.reset_defaults()

'''###################################################################################################'''

# BINARY RANDOM FOREST

# Create a list of the feature columns
feature_cols = ['age', 'sex_F', 'sex_M', 'on_thyroxine_1', 'on_thyroxine_0', 'query_hyperthyroid_1',
                'query_hyperthyroid_0', 'query_hypothyroid_1', 'query_hypothyroid_0', 'pregnant_1',
                'pregnant_0', 'thyroid_surgery_1', 'thyroid_surgery_0', 'I131_treatment_1',
                'I131_treatment_0', 'query_on_thyroxine_1', 'query_on_thyroxine_0', 'on_antithyroid_meds_1',
                'on_antithyroid_meds_0', 'sick_1', 'sick_0', 'tumor_1', 'tumor_0', 'lithium_1', 'lithium_0',
                'goitre_1', 'goitre_0', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'] # ohne TBG!

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
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Performance of the Binary Random Forest model:")
print("Accuracy: {:.3f} \nPrecision: {:.3f} \nRecall: {:.3f} \nF1 score: {:.3f}".format(accuracy, precision, recall, f1))

print('\nBinary Random Forest Classification Report:\n', classification_report(y_test, y_pred))

# Feature Importance Plot
feature_importances = pd.DataFrame(rf.feature_importances_, index=X_train.columns, columns=['importance'])
feature_importances.sort_values(by='importance', ascending=True, inplace=True)
top_10_features = feature_importances.tail(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y=top_10_features.index, data=top_10_features, color='tab:blue')
plt.title('Top 10 Binary Random Forest Feature Importances', fontsize=18)
plt.xlabel('Importance', fontsize=16)
plt.ylabel('Features', fontsize=16)
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('../output/Bin_RF_feature_importance.png', dpi=600)
plt.clf()

# Confusion Matrix Plot
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('Binary RF: Confusion Matrix', fontsize=20)
plt.tight_layout()
plt.savefig('../output/Bin_RF_conf_matrix.png', dpi=600)
plt.clf()
sns.reset_defaults()


'''###################################################################################################'''

print(rf.classes_)
# Calculate the predicted probabilities for the positive class
y_pred_prob_dt = dt.predict_proba(X_test_scale)[:, 0]
y_pred_prob_rf = rf.predict_proba(X_test_scale)[:, 0]

# Generate the False Positive Rate (fpr) and True Positive Rate (tpr) for the ROC curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_prob_dt, pos_label='+')
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf, pos_label='+')

''''# Calculate the AUC-ROC score
auc_roc = roc_auc_score(fpr, tpr)

for in plt.plot(label='ROC Curve (AUC = {:.3f})'.format(auc_roc))

# Print the AUC-ROC score
print("AUC-ROC Score: {:.3f}".format(auc_roc))'''

# Plot the ROC curve
plt.plot(fpr_dt, tpr_dt, label = 'Decision Tree')
plt.plot(fpr_rf, tpr_rf, label = 'Random Forest')
plt.plot([0, 1], [0, 1], color='r', ls='--', label='random\nclassifier')  # Diagonal line for random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('../output/Bin_RF_ROC_AUC.png', dpi=600)
