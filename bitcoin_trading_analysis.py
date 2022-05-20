# Bitcoin Trading strategy - Using Classification-Based Models to Predict Whether to Buy or Sell in the Bitcoin Market
# 1. Problem definition
"""
The problem of predicting a buy or sell signal for a trading strategy is defined in the classification framework,
where the predicted variable has a value of 1 for buy and 0 for sell. This signal is decided through the comparison
of the short-term and long- term price trends.
The data used is from one of the largest bitcoin exchanges in terms of average daily volume, Bitstamp.
The data covers prices from January 2018 to May 2022. Different trend and momentum indicators are created from the data
and are added as features to enhance the performance of the prediction model.

In this case study, the focus will be on:
• Building a trading strategy using classification (classification of long/short signals).
• Feature engineering and constructing technical indicators of trend, momentum, and mean reversion.
• Building a framework for backtesting results of a trading strategy.
• Choosing the right evaluation metric to assess a trading strategy.
"""

# Bitcoin - Loading the data and python packages
# 2.1 Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

from mpl_toolkits.mplot3d import Axes3D

import re
from collections import OrderedDict
from time import time
import sqlite3

from scipy.linalg import svd
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

# Packages for model evaluation and classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Function and modules for data analysis and model evaluation
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression

# Function and modules for deep learning models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Function and modules for time series models
from statsmodels import tsa
#from tsa import arima as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA


import warnings

warnings.filterwarnings ( 'ignore' )

from IPython.html.widgets import interactive, fixed

# Set up display
pd.set_option ( 'display.max_rows', 500)
pd.set_option ( 'display.max_columns', 500)
pd.set_option ( 'display.width', 10000)

# 2. - Loading the Data
# Checking data
dataset = pd.read_csv('bitcoin_bitstamp_data.csv', usecols=['date','open','high','low','close','Volume BTC',
                                                            'Volume USD'], index_col='date',parse_dates=True)

# 3. Exploratory Data Analysis
# 3.1. Descriptive Statistics
# Shape
print ( 'dataset shape \n', dataset.shape)
# peek at data
print ( dataset.tail ( 5 ) )
# peek at data
# describe data
set_option ( 'precision', 3 )
print ( dataset.describe () )

# 4. Data Preparation
# 4.1. Data Cleaning
# Checking for any null values and removing the null values
print ( 'Null Values =', dataset.isnull ().values.any () )
"""
No Null data, we can therefore look at the next step.
"""

# 4.2. Preparing the data for classification
"""
We attach a label to each movement:
 - 1 if the signal is that short term price will go up as compared to the long term.
 - 0 if the signal is that short term price will go down as compared to the long term.
"""
# Initialize the 'signals' DataFrame with the 'signal' column

# Create short simple moving average over the short window
dataset['short_mavg'] = dataset['close'].rolling ( window=10, min_periods=1, center=False ).mean ()

# Create long simple moving average over the long window
dataset['long_mavg'] = dataset['close'].rolling ( window=60, min_periods=1, center=False ).mean ()

# Create signals
dataset['signal'] = np.where ( dataset['short_mavg'] > dataset['long_mavg'], 1.0, 0.0 )

print ( dataset.tail ( 5 ) )

# 4.3. Feature Engineering
"""
We perform feature engineering to construct technical indicators which will be used to make the predictions, and the 
output variable.
The current data of the bitcoin consists of date, open, high, low, close and volume. Using this data we calculate the 
following technical indicators:
 - Moving Average : A moving average provides an indication of the trend of the price movement by cut down the amount 
 of "noise" on a price chart.
 - Stochastic Oscillator %K and %D : A stochastic oscillator is a momentum indicator comparing a particular closing 
 price of a security to a range of its prices over a certain period of time. %K and %D are slow and fast indicators.
 - Relative Strength Index(RSI) :It is a momentum indicator that measures the magnitude of recent price changes to 
 evaluate overbought or oversold conditions in the price of a stock or other asset.
 - Rate Of Change(ROC): It is a momentum oscillator, which measures the percentage change between the current price 
 and the n period past price.
 - Momentum (MOM) : It is the rate of acceleration of a security's price or volume – that is, the speed at which 
 the price is changing.
"""

# Calculation of exponential moving average

def EMA(df, n):
    EMA = pd.Series(df['close'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
    return EMA


dataset['EMA10'] = EMA(dataset, 10)
dataset['EMA30'] = EMA(dataset, 30)
dataset['EMA200'] = EMA(dataset, 200)
print(dataset.tail())

# calculation of rate of change
def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
    return ROC


dataset['ROC10'] = ROC(dataset['close'], 10)
dataset['ROC30'] = ROC(dataset['close'], 30)


# Calculation of price momentum
def MOM(df, n):
    MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
    return MOM


dataset['MOM10'] = MOM(dataset['close'], 10)
dataset['MOM30'] = MOM(dataset['close'], 30)


# Calculation of relative strength index
def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period - 1]] = np.mean(u[:period])  # first value is sum of avg gains
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period])  # first value is sum of avg losses
    d = d.drop(d.index[:(period - 1)])
    rs = u.ewm(com=period - 1, adjust=False).mean() / d.ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + rs)


dataset['RSI10'] = RSI(dataset['close'], 10)
dataset['RSI30'] = RSI(dataset['close'], 30)
dataset['RSI200'] = RSI(dataset['close'], 200)


# Calculation of stochastic oscillator.

def STOK(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    return STOK


def STOD(close, low, high, n):
    STOK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    STOD = STOK.rolling(3).mean()
    return STOD


dataset['%K10'] = STOK(dataset['close'], dataset['low'], dataset['high'], 10)
dataset['%D10'] = STOD(dataset['close'], dataset['low'], dataset['high'], 10)
dataset['%K30'] = STOK(dataset['close'], dataset['low'], dataset['high'], 30)
dataset['%D30'] = STOD(dataset['close'], dataset['low'], dataset['high'], 30)
dataset['%K200'] = STOK(dataset['close'], dataset['low'], dataset['high'], 200)
dataset['%D200'] = STOD(dataset['close'], dataset['low'], dataset['high'], 200)


# Calculation of moving average
def MA(df, n):
    MA = pd.Series(df['close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    return MA


dataset['MA21'] = MA(dataset, 10)
dataset['MA63'] = MA(dataset, 30)
dataset['MA252'] = MA(dataset, 200)
print(dataset.tail())

# Excluding columns that are not needed for our prediction.
dataset = dataset.drop(['high', 'low', 'open', 'Volume USD', 'short_mavg', 'long_mavg'], axis=1)

dataset = dataset.dropna(axis=0)
print(dataset.tail())


# 4.4. Data Visualization

# BTC - Close price
fig, ax = plt.subplots(figsize=(12, 12))
dataset[['close']].plot(grid=True)
plt.legend()
plt.title('BTC - close price')
plt.savefig('BTC - close price.png')
plt.show()

# histograms
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
plt.savefig('BTC - Data visualisation - Histograms.png')
plt.show()


# Signal
"""
Let us look at the distribution of the predicted variable:
"""
plot = dataset.groupby(['signal']).size().plot(kind='barh', color='red')
plt.legend()
plt.savefig('BTC - Signal.png')
plt.show()
"""
The predicted variable is 1 about 35% of the time, meaning there are more sell signals than buy signals. 
"""

# Correlation
correlation = dataset.corr()
plt.figure(figsize=(15,15))
plt.title('Correlation Matrix')
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.savefig('BTC - Correlation heatmap.png')

# 5. Evaluate Algorithms and Models
# 5.1. Train Test Split
"""
We split the dataset into 80% training set and 20% test set.
"""
# split out validation dataset for the end
subset_dataset= dataset.iloc[-10000:]
Y = subset_dataset["signal"]
X = subset_dataset.loc[:, dataset.columns != 'signal']
validation_size = 0.2
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=None)

# 5.2. Test options and evaluation metrics.
"""
Accuracy can be used as the evaluation metric since there is not a significant class imbalance in the data:
"""
# test options for classification
num_folds = 10
scoring = 'accuracy'

# 5.3. Compare models and algorithms.
"""
In order to know which algorithm is best for our strategy, we evaluate the linear, nonlinear, and ensemble models.
"""


# 5.3.1. Models. Checking the classification algorithms:
models = []
models.append(('LR', LogisticRegression(n_jobs=-1)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))

# Neural Network
models.append(('NN', MLPClassifier()))
# Ensemble Models
# Boosting methods
models.append(('AB', AdaBoostClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
# Bagging methods
models.append(('RF', RandomForestClassifier(n_jobs=-1)))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=None)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Cross validation results
fig = plt.figure()
fig.suptitle('BTC - Algorithms Comparison: Kfold results')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
fig.set_size_inches(15,8)
plt.savefig('BTC - Algorithms Comparison: Kfold results.png')
plt.show()

"""
Analysis:
Although some of the models show promising results, we prefer an ensemble model given the huge size of the dataset, 
the large number of features, and an expected non‐ linear relationship between the predicted variable and the features.
Random forest has the best performance among the ensemble models.
"""

# 6. Model tuning and grid-search
# Grid Search: Random Forest Classifier
'''
n_estimators : int (default=100)
    The number of boosting stages to perform. 
    Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
max_depth : integer, optional (default=3)
    maximum depth of the individual regression estimators. 
    The maximum depth limits the number of nodes in the tree. 
    Tune this parameter for best performance; the best value depends on the interaction of the input variables    
criterion : string, optional (default=”gini”)
    The function to measure the quality of a split. 
    Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. 

'''
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
n_estimators = [20, 80]
max_depth = [5, 10]
criterion = ["gini", "entropy"]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
model = RandomForestClassifier(n_jobs=-1)
kfold = KFold(n_splits=num_folds, random_state=None)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

# Print Results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
ranks = grid_result.cv_results_['rank_test_score']
for mean, stdev, param, rank in zip (means, stds, params, ranks):
    print("#%d %f (%f) with: %r" % (rank, mean, stdev, param))

"""
Results:
Best: 0.958482 using {'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 80}
#6 0.926872 (0.016846) with: {'criterion': 'gini', 'max_depth': 5, 'n_estimators': 20}
#5 0.927857 (0.019843) with: {'criterion': 'gini', 'max_depth': 5, 'n_estimators': 80}
#4 0.950578 (0.016277) with: {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 20}
#2 0.956019 (0.014938) with: {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 80}
#8 0.924896 (0.024218) with: {'criterion': 'entropy', 'max_depth': 5, 'n_estimators': 20}
#7 0.925391 (0.026312) with: {'criterion': 'entropy', 'max_depth': 5, 'n_estimators': 80}
#3 0.953061 (0.012572) with: {'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 20}
#1 0.958482 (0.013525) with: {'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 80}
"""
# 7. Finalise the Model
# Finalizing the model with best parameters found during tuning step.
"""
Best: 0.957014 using {'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 80}
"""
# 7.1. Results on the Test Dataset
# prepare model
model = RandomForestClassifier(criterion='entropy', n_estimators=80, max_depth=10, n_jobs=-1) # rbf is default kernel
model.fit(X_train, Y_train)

# estimate accuracy on validation set
predictions = model.predict(X_validation)
print('Accuracy score \n',accuracy_score(Y_validation, predictions))
print('Confusion matrix \n',confusion_matrix(Y_validation, predictions))
print('Classification report \n',classification_report(Y_validation, predictions))

"""
Results:

precision    recall  f1-score   support

         0.0       0.96      0.97      0.96       327
         1.0       0.94      0.93      0.94       179

    accuracy                           0.95       506
   macro avg       0.95      0.95      0.95       506
weighted avg       0.95      0.95      0.95       506

"""
df_cm = pd.DataFrame(confusion_matrix(Y_validation, predictions), columns=np.unique(Y_validation),
                     index = np.unique(Y_validation))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}) # font sizes
plt.show()

# 7.2. Variable Intuition/Feature Importance
"""
Let us look into the Feature Importance of the model
"""

Importance = pd.DataFrame({'Importance':model.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.legend()
plt.savefig('BTC - Variable Importance.png')
plt.show()

# 8. Backtesting Results
# Create column for Strategy Returns by multiplying the daily returns by the position that was held at close of
# business the previous day

backtestdata = pd.DataFrame(index=X_validation.index)
backtestdata['signal_pred'] = predictions
backtestdata['signal_actual'] = Y_validation
backtestdata['Market Returns'] = X_validation['close'].pct_change()
backtestdata['Actual Returns'] = backtestdata['Market Returns'] * backtestdata['signal_actual'].shift(1)
backtestdata['Strategy Returns'] = backtestdata['Market Returns'] * backtestdata['signal_pred'].shift(1)
backtestdata=backtestdata.reset_index()
backtestdata.head()
backtestdata[['Strategy Returns','Actual Returns']].cumsum().hist()
plt.savefig('BTC - Histogram - Strategy Returns vs Actual Returns.png')
plt.show()
backtestdata[['Strategy Returns','Actual Returns']].cumsum().plot()
plt.savefig('BTC - Graph - Strategy Returns vs Actual Returns.png')
plt.show()

# 9. Save Model for Later Use
# Save Model Using Pickle
from pickle import dump
from pickle import load

# save the model to disk
filename = 'bitcoin_finalized_model.sav'
dump(model, open(filename, 'wb'))
# some time later...
# load the model from disk
loaded_model = load(open(filename, 'rb'))
# estimate accuracy on validation set
#rescaledValidationX = scaler.transform(X_validation) #in case the data is scaled.
#predictions = model.predict(rescaledValidationX)
predictions = model.predict(X_validation)
result = mean_squared_error(Y_validation, predictions)
print(result)