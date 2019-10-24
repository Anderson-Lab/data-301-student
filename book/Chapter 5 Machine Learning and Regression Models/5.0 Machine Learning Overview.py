# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 5.0 My three cells: Machine Learning
#
# ## How could we predict the sale price of a home?

# ### Data loading and preprocessing

# +
# %matplotlib inline
import numpy as np
import pandas as pd
pd.options.display.max_rows = 5

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

housing = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/AmesHousing.txt",
                      sep="\t")
housing

train = housing.sample(frac=.5)
val = housing.drop(train.index)

# Features in our model. All quantitative, except Neighborhood.
features = ["Lot Area", "Gr Liv Area",
            "Full Bath", "Half Bath",
            "Bedroom AbvGr", 
            "Year Built", "Yr Sold",
            "Neighborhood"]

X_train_dict = train[features].to_dict(orient="records")
X_val_dict = val[features].to_dict(orient="records")

y_train = train["SalePrice"]
y_val = val["SalePrice"]

# convert categorical variables to dummy variables
vec = DictVectorizer(sparse=False)
vec.fit(X_train_dict)
X_train = vec.transform(X_train_dict)
X_val = vec.transform(X_val_dict)

# standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val)

import autosklearn.regression

automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_regression_example_tmp',
        output_folder='/tmp/autosklearn_regression_example_out',
    )
automl.fit(X_train_sc,y_train)
# -

# ### Machine Learning Black Box (AutoML for regression)

# +
import autosklearn.regression

automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_regression_example_tmp',
        output_folder='/tmp/autosklearn_regression_example_out',
    )
automl.fit(X_train_sc,y_train)
# -

# ### Examine and results

y_val_pred = automl.predict(X_val_sc)
rmse = np.sqrt(((y_val - y_val_pred) ** 2).mean())
print("Our model performance (lower is better), RMSE:",rmse)
median_rmse = np.sqrt(((y_val - np.median(y_train)) ** 2).mean())
print("If we had just guessed the median sale price performance (lower is better), Median-RMSE:",median_rmse)
graph_df = pd.DataFrame({"y":y_val,"y_pred":y_val_pred})
ax = graph_df.plot.scatter(x="y",y="y",alpha=0.5,color='red',legend=True)
ax = graph_df.plot.scatter(x="y",y="y_pred",alpha=0.5,ax=ax,label="Auto ML")
ax.set_xlim([0,600000]);
ax.set_ylim([0,600000]);

# # Exercises
#
# Exercises 1-4 deal with the Titanic data set (`https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv`).

# +
import pandas as pd

titanic_df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv")

titanic_df.head()
# -

# ## Exercise 1
# Using the template provided in the three cells above, load the titanic dataset and construct, evaluate and visualize an automl regression model to predict ``fare`` using ``sex``, ``age``, ``sibsp``, ``parch``, ``embarked`` as the features. What is your `rmse` accuracy? How does that compare to guessing the median?
#
# HINT: You will have to deal with missing values.

# +
# YOUR CODE HERE
# BEGIN SOLUTION
import pandas as pd

titanic_df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv")

train = titanic_df.sample(frac=.5)
val = titanic_df.drop(train.index)

# Features in our model. All quantitative, except Neighborhood.
features = ["sex","age","sibsp","parch","embarked"]

X_train_dict = train[features].to_dict(orient="records")
X_val_dict = val[features].to_dict(orient="records")

y_train = train["fare"]
y_val = val["fare"]

# convert categorical variables to dummy variables
vec = DictVectorizer(sparse=False)
vec.fit(X_train_dict)
X_train = pd.DataFrame(vec.transform(X_train_dict)).fillna(0)
X_val = pd.DataFrame(vec.transform(X_val_dict)).fillna(0)

# standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val)
# END SOLUTION

# +
import autosklearn.regression

automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_regression_example_tmp2',
        output_folder='/tmp/autosklearn_regression_example_out2',
    )
automl.fit(X_train_sc,y_train)
# -

y_val_pred = automl.predict(pd.DataFrame(X_val_sc).fillna(0))
rmse = np.sqrt(((y_val - y_val_pred) ** 2).mean())
print("Our model performance (lower is better), RMSE:",rmse)
median_rmse = np.sqrt(((y_val - np.median(y_train)) ** 2).mean())
print("If we had just guessed the median fare performance (lower is better), Median-RMSE:",median_rmse)
graph_df = pd.DataFrame({"y":y_val,"y_pred":y_val_pred})
ax = graph_df.plot.scatter(x="y",y="y",alpha=0.5,color='red',legend=True)
ax = graph_df.plot.scatter(x="y",y="y_pred",alpha=0.5,ax=ax,label="Auto ML")
#ax.set_xlim([0,600000]);
#ax.set_ylim([0,600000]);

# ## Exercise 2
# Repeat the same experiment, but see if including another feature or two makes any difference. Why did you pick that feature?

# +
# YOUR CODE HERE
# BEGIN SOLUTION
import pandas as pd

titanic_df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv")

train = titanic_df.sample(frac=.5)
val = titanic_df.drop(train.index)

# Features in our model. All quantitative, except Neighborhood.
features = ["sex","age","sibsp","parch","embarked","cabin","body"]

X_train_dict = train[features].to_dict(orient="records")
X_val_dict = val[features].to_dict(orient="records")

y_train = train["fare"]
y_val = val["fare"]

# convert categorical variables to dummy variables
vec = DictVectorizer(sparse=False)
vec.fit(X_train_dict)
X_train = pd.DataFrame(vec.transform(X_train_dict)).fillna(0)
X_val = pd.DataFrame(vec.transform(X_val_dict)).fillna(0)

# standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val)

import autosklearn.regression

automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_regression_example_tmp3',
        output_folder='/tmp/autosklearn_regression_example_out3',
    )
automl.fit(X_train_sc,y_train)

y_val_pred = automl.predict(pd.DataFrame(X_val_sc).fillna(0))
rmse = np.sqrt(((y_val - y_val_pred) ** 2).mean())
print("Our model performance (lower is better), RMSE:",rmse)
median_rmse = np.sqrt(((y_val - np.median(y_train)) ** 2).mean())
print("If we had just guessed the median fare performance (lower is better), Median-RMSE:",median_rmse)
graph_df = pd.DataFrame({"y":y_val,"y_pred":y_val_pred})
ax = graph_df.plot.scatter(x="y",y="y",alpha=0.5,color='red',legend=True)
ax = graph_df.plot.scatter(x="y",y="y_pred",alpha=0.5,ax=ax,label="Auto ML")
#ax.set_xlim([0,600000]);
#ax.set_ylim([0,600000]);
# END SOLUTION
# -

# ## Exercise 3
# Now let's switch to the classification problem of predicting ``survived``. You can use any features you want (as long as there are a mixture of categorical and numeric). You can find the documentation for Auto ML classification at https://automl.github.io/auto-sklearn/master/#manual. Report the accuracy of your model on the validation dataset. Is this better than guessing?

# +
# YOUR CODE HERE
# BEGIN SOLUTION
import pandas as pd

titanic_df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv")

train = titanic_df.sample(frac=.5)
val = titanic_df.drop(train.index)

# Features in our model. All quantitative, except Neighborhood.
features = ["sex","age","sibsp","parch","embarked","cabin","body","fare"]

X_train_dict = train[features].to_dict(orient="records")
X_val_dict = val[features].to_dict(orient="records")

y_train = train["survived"]
y_val = val["survived"]

# convert categorical variables to dummy variables
vec = DictVectorizer(sparse=False)
vec.fit(X_train_dict)
X_train = pd.DataFrame(vec.transform(X_train_dict)).fillna(0)
X_val = pd.DataFrame(vec.transform(X_val_dict)).fillna(0)

# standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val)

import autosklearn.classification

automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train_sc,y_train)

y_val_pred = automl.predict(pd.DataFrame(X_val_sc).fillna(0))
import sklearn.metrics
acc = sklearn.metrics.accuracy_score(y_val, y_val_pred)
print("Our model performance (higher is better), Accuracy:",acc)
#print("If we had just guessed the median fare performance (lower is better), Median-RMSE:",median_rmse)
#graph_df = pd.DataFrame({"y":y_val,"y_pred":y_val_pred})
#ax = graph_df.plot.scatter(x="y",y="y",alpha=0.5,color='red',legend=True)
#ax = graph_df.plot.scatter(x="y",y="y_pred",alpha=0.5,ax=ax,label="Auto ML")
#ax.set_xlim([0,600000]);
#ax.set_ylim([0,600000]);
# END SOLUTION
# -


