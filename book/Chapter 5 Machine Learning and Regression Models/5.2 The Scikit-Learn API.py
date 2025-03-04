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

# # 5.2 The Scikit-Learn API
#
# In the previous section, we implemented $k$-nearest neighbors from scratch. Now we will see how to implement it using [Scikit-Learn](http://scikit-learn.org/), a Python library that makes it easy to train and use machine learning models. All models are trained and used in the exact same way:
#
# 1. Declare the model.
# 2. Fit the model to training data, consisting of both features $X$ and labels $y$.
# 3. Use the model to predict the labels for new values of the features.
#
# Let's take a look at how we would use this API to train a model on the Ames housing data set to predict the 2011 price of the Old Town house from the previous section. Scikit-Learn assumes that the data has already been completely converted to  quantitative variables and that the variables have already been standardized (if desired). The code below reads in the data and does the necessary preprocessing. 
#
# (All of this code is copied from the previous section. Read the code, and if you are not sure what a particular line does, refer back to the previous section.)

# +
# %matplotlib inline
import numpy as np
import pandas as pd
pd.options.display.max_rows = 5

housing = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/AmesHousing.txt",
                      sep="\t")

housing["Date Sold"] = housing["Yr Sold"] + housing["Mo Sold"] / 12
features = ["Lot Area", "Gr Liv Area",
            "Full Bath", "Half Bath",
            "Bedroom AbvGr", 
            "Year Built", "Date Sold",
            "Neighborhood"]
X_train = pd.get_dummies(housing[features])
y_train = housing["SalePrice"]

x_new = pd.Series(index=X_train.columns)
x_new["Lot Area"] = 9000
x_new["Gr Liv Area"] = 1400
x_new["Full Bath"] = 2
x_new["Half Bath"] = 1
x_new["Bedroom AbvGr"] = 3
x_new["Year Built"] = 1980
x_new["Date Sold"] = 2011
x_new["Neighborhood_OldTown"] = 1
x_new.fillna(0, inplace=True)

X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train_sc = (X_train - X_train_mean) / X_train_std
x_new_sc = (x_new - X_train_mean) / X_train_std

X_train_sc
# -

# `X_train_sc` is a matrix of all numbers, which is the form that Scikit-Learn expects. Now let's see how to use Scikit-Learn to fit a $k$-nearest neighbors model to this data.

# +
from sklearn.neighbors import KNeighborsRegressor

# Step 1: Declare the model.
model = KNeighborsRegressor(n_neighbors=30)

# Step 2: Fit the model to training data.
model.fit(X_train_sc, y_train)

# Step 3: Use the model to predict for new observations.
# Scikit-Learn expects 2-dimensional arrays, so we need to 
# turn the Series into a DataFrame with 1 row.
X_new_sc = x_new_sc.to_frame().T
model.predict(X_new_sc)
# -

# This is the exact same prediction that we got by implementing $k$-nearest neighbors manually. 
#
# In the case of training a machine learning model to predict for a single observation, Scikit-Learn may seem like overkill. In fact, the above Scikit-Learn code was 5 lines, whereas our implementation of $k$-nearest neighbors in the previous section was only 4 lines. However, learning Scikit-Learn will pay off as the problems become more complex.

# ## Preprocessing in Scikit-Learn
#
# We constructed `X_train_sc` and `x_new_sc` above using just basic `pandas` operations. But it is also possible to have Scikit-Learn do this preprocessing for us. The preprocessing objects in Scikit-Learn all follow the same basic pattern:
#
# 1. First, the preprocessing object has to be "fit" to a data set.
# 2. The `.transform()` method actually processes the data. This method can be called repeatedly on multiple data sets and is guaranteed to process each data set in exactly the same way.
#
# It might not be obvious why it is necessary to first "fit" the preprocessing object to a data set before using it to process data. Hopefully, the following examples will make this clear.
#
# ### Example 1: Dummy Encoding
#
# Instead of using `pd.get_dummies()`, we can do dummy encoding in Scikit-Learn using the `DictVectorizer` tool. There is one catch: `DictVectorizer` expects the data as a list of dictionaries, not as a `DataFrame`. But each row of a `DataFrame` can be represented as a dictionary, where the keys are the column names and the values are the data. `Pandas` provides a convenience function, `.to_dict()`, that converts a `DataFrame` into a list of dictionaries.

X_train_dict = housing[features].to_dict(orient="records")
X_train_dict[:2]

# Now we pass this list to `DictVectorizer`, which will expand each categorical variable (e.g., "Neighborhood") into dummy variables. When the vectorizer is fit to the training data, it will learn all of the possible categories for each categorical variable so that when `.transform()` is called on different data sets, the same dummy variables will be returned (and in the same order). This is important for us because we need to apply the encoding to two data sets, the training data and the new observation, and we want to be sure that the same dummy variables appear in both.

# +
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer(sparse=False)
vec.fit(X_train_dict)

X_train = vec.transform(X_train_dict)
x_new_dict = {
    "Lot Area": 9000,
    "Gr Liv Area": 1400,
    "Full Bath": 2,
    "Half Bath": 1,
    "Bedroom AbvGr": 3,
    "Year Built": 1980,
    "Date Sold": 2011,
    "Neighborhood": "OldTown"
}
X_new = vec.transform([x_new_dict])

X_train
# -

# ### Example 2: Scaling
#
# We can also use Scikit-Learn to scale our data. The `StandardScaler` function standardizes data, but there are other functions, such as `Normalizer` and `MinMaxScaler`, that normalize and apply min-max scaling to the data, respectively. 
#
# In the previous section, we standardized both the training data and the new observation with respect to the _training data_. To specify that the standardization should be with respect to the training data, we fit the scaler to the training data. Then, we use the scaler to transform both the training data and the new observation.

# +
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_sc = scaler.transform(X_train)
X_new_sc = scaler.transform(X_new)

X_train_sc
# -

# ## Putting It All Together
#
# The following example shows a complete pipeline: from reading in the raw data and processing it, to fitting a machine learning model and using it for prediction.

# +
# Read in the data.
housing = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/AmesHousing.txt", 
                      sep="\t")

# Define the features.
housing["Date Sold"] = housing["Yr Sold"] + housing["Mo Sold"] / 12
features = ["Lot Area", "Gr Liv Area",
            "Full Bath", "Half Bath",
            "Bedroom AbvGr", 
            "Year Built", "Date Sold",
            "Neighborhood"]

# Define the training data.
# Represent the features as a list of dicts.
X_train_dict = housing[features].to_dict(orient="records")
X_new_dict = [{
    "Lot Area": 9000,
    "Gr Liv Area": 1400,
    "Full Bath": 2,
    "Half Bath": 1,
    "Bedroom AbvGr": 3,
    "Year Built": 1980,
    "Date Sold": 2011,
    "Neighborhood": "OldTown"
}]
y_train = housing["SalePrice"]

# Dummy encoding
vec = DictVectorizer(sparse=False)
vec.fit(X_train_dict)
X_train = vec.transform(X_train_dict)
X_new = vec.transform(X_new_dict)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_new_sc = scaler.transform(X_new)

# K-Nearest Neighbors Model
model = KNeighborsRegressor(n_neighbors=30)
model.fit(X_train_sc, y_train)
model.predict(X_new_sc)
# -

# # Exercises

# **Exercise 1.** (This exercise is identical to Exercise 2 from the previous section, except it asks you to use Scikit-Learn.) You would like to predict how much a male diner will tip on a bill of \\$40.00 on a Sunday. Use Scikit-Learn to build a $k$-nearest neighbors model to answer this question, using the Tips dataset (`https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv`) as your training data.

# +
# TYPE YOUR CODE HERE
# BEGIN SOLUTION
# Read in the data.
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

tips = pd.read_csv('https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv')

# Define the features.
features = ["total_bill","sex","day"]

# Define the training data.
# Represent the features as a list of dicts.
X_train_dict = tips[features].to_dict(orient="records")
X_new_dict = [{
    "total_bill":40,
    "sex":"Male",
    "day":"Sun"
}]
y_train = tips["tip"]

# Dummy encoding
vec = DictVectorizer(sparse=False)
vec.fit(X_train_dict)
X_train = vec.transform(X_train_dict)
X_new = vec.transform(X_new_dict)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_new_sc = scaler.transform(X_new)

# K-Nearest Neighbors Model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train_sc, y_train)
display(model.predict(X_new_sc))

model = KNeighborsRegressor(n_neighbors=10)
model.fit(X_train_sc, y_train)
display(model.predict(X_new_sc))

model = KNeighborsRegressor(n_neighbors=20)
model.fit(X_train_sc, y_train)
display(model.predict(X_new_sc))
# END SOLUTION
# -


