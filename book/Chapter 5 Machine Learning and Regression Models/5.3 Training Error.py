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

# # 5.3 Training Error
#
# In the previous sections, we learned to build regression models. In this section, we will learn one way to evaluate the quality of a regression model: the training error. We will also discuss the shortcomings of using training error to measure the quality of a regression model.

# +
# %matplotlib inline
import numpy as np
import pandas as pd
pd.options.display.max_rows = 5

housing = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/AmesHousing.txt",
                      sep="\t")
housing
# -

# ## Performance Metrics for Regression Models
#
# To evaluate the performance of a regression model, we compare the predicted labels from the model against the true labels. Since the labels are quantitative, it makes sense to look at the difference between each predicted label $\hat y_i$ and the true label $y_i$. 
#
# One way to make sense of these differences is to square each difference and average the squared differences. This measure of error is known as **mean squared error** (or **MSE**, for short):
#
# $$ 
# \begin{align*}
# \textrm{MSE} &= \textrm{mean of } (y_i - \hat y_i)^2.
# \end{align*}
# $$ 
#
# MSE is difficult to interpret because its units are the square of the units of $y$. To make MSE more interpretable, it is common to take the _square root_ of the MSE to obtain the **root mean squared error** (or RMSE, for short):
#
# $$ 
# \begin{align*}
# \textrm{RMSE} &= \sqrt{\textrm{MSE}}.
# \end{align*}
# $$ 
#
# The RMSE measures how off a "typical" prediction is. Notice that the reasoning above is exactly the same reasoning that we used in Chapter 1 when we defined the variance and the standard deviation.
#
# Another common measure of error is the **mean absolute error** (or **MAE**, for short):
#
# $$ 
# \begin{align*}
# \textrm{MAE} &= \textrm{mean of } |y_i - \hat y_i|.
# \end{align*}
# $$ 
#
# Like the RMSE, the MAE measures how off a "typical" prediction is. There are other metrics that can be used to measure the quality of a regression model, but these are the most common ones.

# ## Training Error
#
# To calculate the MSE, RMSE, or MAE, we need data where the true labels are known. Where do we find such data? One natural source of labeled data is the training data, since we needed the true labels to be able to train a model.
#
# For a $k$-nearest neighbors model, the training data is the data from which the $k$-nearest neighbors are selected. So to calculate the training RMSE, we do the following:
#
# For each observation in the training data:
# 1. Find its $k$-nearest neighbors in the training data.
# 2. Average the labels of the $k$-nearest neighbors to obtain the predicted label.
# 3. Subtract the predicted label from the true label.
#
# At this point, we can average the square of these differences to obtain the MSE or average their absolute values to obtain the MAE.
#
# Let's calculate the training MSE for a 10-nearest neighbors model for house price using a subset of features from the Ames housing data set. First, we extract the variables that we need.

# +
# Features in our model. All quantitative, except Neighborhood.
features = ["Lot Area", "Gr Liv Area",
            "Full Bath", "Half Bath",
            "Bedroom AbvGr", 
            "Year Built", "Yr Sold",
            "Neighborhood"]

X_train_dict = housing[features].to_dict(orient="records")
y_train = housing["SalePrice"]
# -

# Now we will use Scikit-Learn to preprocess the features...

# +
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

vec = DictVectorizer(sparse=False)
vec.fit(X_train_dict)
X_train = vec.transform(X_train_dict)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
# -

# ...and to fit the $k$-nearest neighbors model to the data.

# +
from sklearn.neighbors import KNeighborsRegressor

# Fit a 10-nearest neighbors model.
model = KNeighborsRegressor(n_neighbors=10)
model.fit(X_train_sc, y_train)

# Calculate the model predictions on the training data.
y_train_pred = model.predict(X_train_sc)
y_train_pred
# -

# Now it's time to compare these predictions to the true labels, which we know, since this is the training data.

# Calculate the mean-squared error.
mse = ((y_train - y_train_pred) ** 2).mean()
mse

# This number is very large and not very interpretable (because it is in units of "dollars squared"). Let's take the square root to obtain the RMSE.

rmse = np.sqrt(mse)
rmse

# The RMSE says that our model's predictions are, on average, off by about \\$33,000. This is not great, but it is also not too bad when an average house is worth about \\$180,000.

# ## The Problem with Training Error
#
# Training error is not a great measure of the quality of a model. To see why, consider a 1-nearest neighbor regression model. Before you read on, can you guess what the training error of a 1-nearest neighbor regression model will be?

# +
# Fit a 1-nearest neighbors model.
model = KNeighborsRegressor(n_neighbors=1)
model.fit(X_train_sc, y_train)

# Calculate the model predictions on the training data.
y_train_pred = model.predict(X_train_sc)

# Calculate the MAE
(y_train - y_train_pred).abs().mean()
# -

# The training error of this model seems too good to be true. Can our model really be off by just \$57.85 on average?
#
# The error is so small because the nearest neighbor to any observation in the training data will be the observation itself! In fact, if we look at the vector of differences between the true and predicted labels, we see that most of the differences are zero.

y_train - y_train_pred

# Why isn't the MSE exactly equal to 0, then? That is because there may be multiple houses in the training data with the exact same values for all of the features, so there may be multiple observations that are a distance of 0.0 away. Any one of these observations has equal claim to being the "1-nearest neighbor". If we happen to select one of the _other_ houses in the training data as the nearest neighbor, then its price will in general be different.
#
# How many predictions did the 1-nearest neighbor model get wrong?

(y_train != y_train_pred).sum()

# The 1-nearest neighbor model nailed the price exactly for all but 22 of the 2930 houses, so the training error is small.
#
# Of course, a 1-nearest neighbor is unlikely to be the best model for predicting house prices. If one house in the training data happened to cost \\$10,000,000, it would not be sensible to predict another house to cost \\$10,000,000 -- even one very similar to it. This is why we usually average over multiple neighbors (i.e., $k$ neighbors) to make predictions.  
#
# In the next section, we will learn a better way to measure the quality of a model than training error.

# # Exercises

# **Exercise 1.** Using the Tips data set (`https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv`), train $k$-nearest neighbors regression models to predict the tip for different values of $k$. Calculate the training MAE of each model and make a plot showing this training error as a function of $k$.

# +
# TYPE YOUR CODE HERE.
# BEGIN SOLUTION
tips = pd.read_csv('https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv')

# Define the features.
features = ["total_bill","sex"]

# Define the training data.
# Represent the features as a list of dicts.
X_train_dict = tips[features].to_dict(orient="records")
X_new_dict = [{
    "total_bill":40,
    "sex":"Male"
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

results = pd.DataFrame(columns=["MAE"],index=[3,10,20,30])

# K-Nearest Neighbors Model
for k in results.index:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train_sc, y_train)
    y_train_pred = model.predict(X_train_sc)

    # Calculate the MAE
    results.loc[k]["MAE"] = (y_train - y_train_pred).abs().mean()
    
results["MAE"].plot.bar()
# END SOLUTION
# -


