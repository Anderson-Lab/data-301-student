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

# # 5.5 Model Selection and Hyperparameter Tuning
#
# This section will use the tools developed in the previous section to answer two important questions:
#
# - Model Selection: How do we determine which model is best?
# - Hyperparameter Tuning: How do we choose hyperparameters, such as $k$ in $k$-nearest neighbors?
#
# In the previous section, we saw how to use training and validation sets to estimate how well the model will perform on future data. A natural way to decide between competing models (or hyperparameters) is to choose the one that minimizes the validation error.

# +
# %matplotlib inline
import numpy as np
import pandas as pd
pd.options.display.max_rows = 5

housing = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/AmesHousing.txt",
                      sep="\t")
housing
# -

# ## $K$-Fold Cross Validation
#
# Previously, we carried out cross validation by splitting the data into 2 halves, alternately using one half to train the model and the other to evaluate the model. In general, we can split the data into $k$ subsamples, alternately training the data on $k-1$ subsamples and evaluating the model on the $1$ remaining subsample, i.e., the validation set. This produces $k$ somewhat independent estimates of the test error. This procedure is known as **$k$-fold cross validation**. (Be careful not to confuse the $k$ in $k$-fold cross validation with the $k$ in $k$-nearest neighbors.) Therefore, the specific version of cross validation that we saw earlier is $2$-fold cross validation.
#
# A schematic of $4$-fold cross validation is shown below.
#
# ![](k-folds.png)
#
# Implementing $k$-fold cross validation from scratch for $k > 2$ is straightforward but messy, so we will usually let Scikit-Learn do it for us.

# ## Cross Validation in Scikit-Learn
#
# Scikit-Learn provides a function, `cross_val_score`, that will carry out all aspects of $k$-fold cross validation: 
#
# 1. split the data into $k$ subsamples
# 2. combine the first $k-1$ subsamples into a training set and train the model
# 3. evaluate the model predictions on the last ($k$th) held-out subsample
# 4. repeat steps 2-3 $k$ times (i.e. $k$ "folds"), each time holding out a different one of the $k$ subsamples
# 4. calculate $k$ "scores", one from each validation set
#
# There is one subtlety to keep in mind. Training a $k$-nearest neighbors model is not just about fitting the model; it also involves dummifying the categorical variables and scaling the variables. These preprocessing steps should be included in the cross-validation process. They cannot be done ahead of time.
#
# For example, suppose we run $5$-fold cross validation. Then:
#
# - When subsamples 1-4 are used for training and subsample 5 for validation, the observations have to be standardized with respect to the mean and SD of subsamples 1-4.
# - When subsamples 2-5 are used for training and subsample 1 for validation, the observations have to be standardized with respect to the mean and SD of subsamples 2-5.
# - And so on.
#
# We cannot simply standardize all of the data once at the beginning and run cross validation on the standardized data. To do so would be allowing the model to peek at the validation set during training. That's because each training set would be standardized with respect to a mean and SD that is calculated from all data, including the validation set. To be completely above board, we should standardize each training set with respect to the mean and SD of just that training set.
#
# Fortunately, Scikit-Learn provides a `Pipeline` object that allows us to chain these preprocessing steps together with the model we want to fit.

# +
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

# get the features (in dict format) and the labels
# (do not split into training and validation sets)
features = ["Lot Area", "Gr Liv Area",
            "Full Bath", "Half Bath",
            "Bedroom AbvGr", 
            "Year Built", "Yr Sold",
            "Neighborhood"]
X_dict = housing[features].to_dict(orient="records")
y = housing["SalePrice"]

# specify the pipeline
vec = DictVectorizer(sparse=False)
scaler = StandardScaler()
model = KNeighborsRegressor(n_neighbors=10)
pipeline = Pipeline([("vectorizer", vec), ("scaler", scaler), ("fit", model)])
# -

# This entire `Pipeline` can be passed to `cross_val_score`, along with the data, the number of folds $k$ (`cv`), and the type of score (`scoring`). So $5$-fold cross validation in Scikit-Learn would look as follows:

pipeline.fit(X_dict,y)
pipeline.predict(X_dict)

# +
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X_dict, y, 
                         cv=5, scoring="neg_mean_squared_error")
scores
# -

# Notice that we get five (negative) validation MSEs, one from each of the 5 folds. `cross_val_score` returns the _negative_ MSE, instead of the MSE, because by definition, a _higher_ score is better. (Since we want the MSE to be as _low_ as possible, we want the negative MSE to be as _high_ as possible.)
#
# To come up with a single overall estimate of the test MSE, we flip the signs and average the MSEs:

np.mean(-scores)

# The RMSE is the square root of the MSE:

np.sqrt(np.mean(-scores))

# ## Hyperparameter Tuning
#
# How do we choose $k$? We can simply try all values of $k$ and pick the one with the smallest (test) MSE.

# +
vec = DictVectorizer(sparse=False)
scaler = StandardScaler()

# calculates estimate of test error based on 10-fold cross validation
def get_cv_error(k):
    model = KNeighborsRegressor(n_neighbors=k)
    pipeline = Pipeline([("vectorizer", vec), ("scaler", scaler), ("fit", model)])
    mse = np.mean(-cross_val_score(
        pipeline, X_dict, y, 
        cv=10, scoring="neg_mean_squared_error"
    ))
    return mse
    
ks = pd.Series(range(1, 51))
#ks.index = range(1, 51)
test_errs = ks.apply(get_cv_error)

test_errs.plot.line()
test_errs.sort_values()
# -

# The MSE is minimized near $k = 4$, which suggests that a $4$-nearest neighbors model is optimal for prediction.

# ## Model Selection
#
# Suppose we are not sure whether `Yr Sold` should be included in the $4$-nearest neighbors model or not. To determine whether or not it should be included, we can fit a model with `Yr Sold` included and another model with it excluded, and see which model has the better (test) MSE.

vec = DictVectorizer(sparse=False)
scaler = StandardScaler()
model = KNeighborsRegressor(n_neighbors=4)
pipeline = Pipeline([("vectorizer", vec), ("scaler", scaler), ("fit", model)])

features = ["Lot Area", "Gr Liv Area",
            "Full Bath", "Half Bath",
            "Bedroom AbvGr", 
            "Year Built", "Yr Sold",
            "Neighborhood"]
X_dict = housing[features].to_dict(orient="records")
np.mean(
    -cross_val_score(pipeline, X_dict, y, cv=10, scoring="neg_mean_squared_error")
)

features = ["Lot Area", "Gr Liv Area",
            "Full Bath", "Half Bath",
            "Bedroom AbvGr", 
            "Year Built",
            "Neighborhood"]
X_dict = housing[features].to_dict(orient="records")
-cross_val_score(pipeline, X_dict, y, cv=10, scoring="neg_mean_squared_error").mean()

# The MSE actually goes down when we remove `Yr Sold`, so it seems that the model is better off without this variable.

# ## GridSearchCV
# Now it is time to improve our skills with help from sklearn. We could continue to use apply and cross_val_score for different hyperparameters and different methods, but with help from sklearn, our role is a lot more streamlined.
#
# Our example will construct a pipeline that does dimensionality reduction followed by regression with KNN. Unsupervised PCA and dimensionality reductions are compared to univariate feature selection during the grid search.

# +
# Some structure pulled from authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsRegressor

pipe = Pipeline([
    ('vec',DictVectorizer(sparse=False)),
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', PCA()),
    ('regression', KNeighborsRegressor())
])

features = ["Lot Area", "Gr Liv Area",
            "Full Bath", "Half Bath",
            "Bedroom AbvGr", 
            "Year Built",
            "Neighborhood"]
X_dict = housing[features].to_dict(orient="records")

## GRID SEARCH
N_FEATURES_OPTIONS = [2, 4, 8]
k_OPTIONS = [1, 10, 100]
param_grid = {
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'regression__n_neighbors': k_OPTIONS
}

grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid, iid=False,scoring='neg_mean_squared_error',return_train_score=True)
grid.fit(X_dict, y)
# -

grid.cv_results_

# +
mean_scores = np.array(grid.cv_results_['mean_test_score'])

# scores are in the order of param_grid iteration, which is alphabetical
#mean_scores = mean_scores.reshape(len(k_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# -

results = pd.DataFrame(grid.cv_results_['params'])
results["mean_test_score"] = mean_scores

results

results.pivot_table(index="regression__n_neighbors",columns="reduce_dim__n_components",values="mean_test_score").plot.bar()

# ## Example below shows how to write a custom function transformer

# +

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

pipe = Pipeline([
    ('pre',FunctionTransformer(validate=False)),
    ('vec',DictVectorizer(sparse=False)),
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', PCA()),
    ('regression', KNeighborsRegressor())
])

## GRID SEARCH

def func_features(df,features):
    X_dict = df[features].to_dict(orient="records")
    return X_dict

FUNC_OPTIONS = [
    lambda df: func_features(df,["Lot Area", "Gr Liv Area","Full Bath", "Half Bath","Bedroom AbvGr", "Year Built","Neighborhood"]),
    lambda df: func_features(df,["Lot Area", "Gr Liv Area","Full Bath", "Half Bath","Bedroom AbvGr","Neighborhood"]),
]

#a = FunctionTransformer(func=FUNC_OPTIONS[0],validate=False)
#a.fit(housing)

N_FEATURES_OPTIONS = [2, 4, 8]
k_OPTIONS = [1, 10, 100]
param_grid = {
        'pre__func': FUNC_OPTIONS,
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'regression__n_neighbors': k_OPTIONS
}

grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid, iid=False,scoring='neg_mean_squared_error',return_train_score=True)
grid.fit(housing, y)
# -

grid.cv_results_

# # Exercises

# # Exercise 1
# Perform a grid search to help you find the best parameters for predicting the ``tip`` from the ``tips`` dataset. It is up to you to select the features and methods we've used in this book; however, you must include at least one categorical feature and you must include at least three steps in your pipeline like the one used above. Further, you must try parameters for at least 2 of the steps in the pipeline (i.e., you don't have to tune the DictVectorizer). Finally, pull it all together by trying different scaling methods as part of your analysis.


