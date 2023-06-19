import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

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


# Loop through all columns with binary values and one-hot encode them
for col in data.columns:
    if data[col].dtype == 'object' and set(data[col].unique()) == {'t', 'f'}:
        data[col] = data[col].apply(lambda x: 1 if x == 't' else 0)
        data = pd.get_dummies(data, columns=[col])

# only go for thyroid disease or no thyroid disease
data.loc[(data['target'] == '1 - hyperthyroid conditions') | (data['target'] == '2 - hypothyroid conditions'), 'target'] = '+'
data.loc[(data['target'] == '3 - binding protein') | (data['target'] == '4 - general health') |
         (data['target'] == '5 - replacement therapy') | (data['target'] == '6 - antithyroid treatment') |
         (data['target'] == '7 - miscellaneous'), 'target'] = '-'

"#################SVM##############################################"
# SVM
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
k_values = range(10, 20)  #### adjust here computational cost ##### ###best value is 17
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
