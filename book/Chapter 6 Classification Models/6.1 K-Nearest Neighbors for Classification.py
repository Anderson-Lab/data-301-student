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

# # 6.1 K-Nearest Neighbors for Classification
#
# _Classification models_ are used when the label we want to predict is categorical. In this section, we will train a classification model to predict the color of a wine (red or white) from its chemical properties. 
#
# The training data for the red and white wines are stored in separate files on Github (https://github.com/dlsun/data-science-book/tree/master/data/wines). Let's read in the two datasets, add a column for the color ("red" or "white"), and combine them into one `DataFrame`.

# +
# %matplotlib inline
import numpy as np
import pandas as pd
pd.options.display.max_rows = 5

reds = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/wines/reds.csv",
                   sep=";")
whites = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/wines/whites.csv", 
                     sep=";")

reds["color"] = "red"
whites["color"] = "white"

wines = pd.concat([reds, whites], 
                  ignore_index=True)
wines
# -

# Let's focus on just two features for now: volatile acidity and total sulfur dioxide. Let's plot the training data, using color to represent the class label.

# +
colors = wines["color"].map({
    "red": "darkred",
    "white": "gold"
})

wines.plot.scatter(
    x="volatile acidity", y="total sulfur dioxide", c=colors, 
    alpha=.3, xlim=(0, 1.6), ylim=(0, 400)
);
# -

# Now suppose that we have a new wine with volatile acidity .85 and total sulfur dioxide 120, represented by a black circle in the plot below. Is this likely a red wine or a white wine?
#
# ![](classification.png)
#
# It is not hard to guess that this wine is probably red, just by looking at the plot. The reasoning goes like this: most of the wines in the training data that were "close" to this wine were red, so it makes sense to predict that this wine is also red. This is precisely the idea behind the $k$-nearest neighbors classifier:
#
# 1. Calculate the distance between the new point and each point in the training data, using some distance metric on the features.
# 2. Determine the $k$ closest points. Of these $k$ closest points, count up how many of each class label there are.
# 3. The predicted class of the new point is whichever class was most common among the $k$ closest points.
#
# The only difference between the $k$-nearest neighbors classifier and the $k$-nearest neighbors regressor from the previous chapter is the last step. Instead of averaging the labels of the $k$ neighbors to obtain our prediction, we count up the number of occurrences of each category among the labels and take the most common one. It makes sense that we have to do something different because the label is now categorical instead of quantitative. This is yet another example of the general principle that was introduced in Chapter 1: the analysis changes depending on the variable type!

# # Implementing the K-Nearest Neighbors Classifier
#
# Let's implement $9$-nearest neighbors for the wine above. First, we extract the training data and scale the features:

# +
X_train = wines[["volatile acidity", "total sulfur dioxide"]]
y_train = wines["color"]

X_train_sc = (X_train - X_train.mean()) / X_train.std()
# -

# Then, we create a `Series` for the new wine, being sure to scale it in the exact same way:

# +
x_new = pd.Series()
x_new["volatile acidity"] = 0.85
x_new["total sulfur dioxide"] = 120

x_new_sc = (x_new - X_train.mean()) / X_train.std()
x_new_sc
# -

# Now we calculate the (Euclidean) distance between this new wine and each wine in the training data, and sort the distances from smallest to largest.

dists = np.sqrt(((X_train_sc - x_new_sc) ** 2).sum(axis=1))
dists_sorted = dists.sort_values()
dists_sorted

# The first 9 entries of this sorted list will be the 9 nearest neighbors. Let's get their index.

inds_nearest = dists_sorted.index[:9]
inds_nearest

# Now we can look up these indices in the original data.

wines.loc[inds_nearest]

# As a sanity check, notice that these wines are all similar to the new wine in terms of volatile acidity and total sulfur dioxide. To make a prediction for this new wine, we need to count up how many reds and whites there are among these 9-nearest neighbors.

wines.loc[inds_nearest, "color"].value_counts()

# There were more reds than whites, so the 9-nearest neighbors model predicts that the wine is red.
#
# As a measure of confidence in a prediction, classification models often report the predicted _probability_ of each label, instead of just the predicted label. The predicted probability of a class in a $k$-nearest neighbors model is simply the proportion of the $k$ neighbors that are in that class. In the example above, instead of simply predicting that the wine is red, we could have instead said that the wine has a $6/9 = .667$ probability of being red.

# # K-Nearest Neighbors Classifier in Scikit-Learn
#
# Now let's see how to implement the same $9$-nearest neighbors model above using Scikit-Learn.

# +
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# define the training data
X_train = wines[["volatile acidity", "total sulfur dioxide"]]
y_train = wines["color"]

X_train_sc = (X_train - X_train.mean()) / X_train.std()

# standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)

# fit the 9-nearest neighbors model
model = KNeighborsClassifier(n_neighbors=9)
model.fit(X_train_sc, y_train)

# define the test data (Scikit-Learn expects a matrix)
x_new = pd.DataFrame()
x_new["volatile acidity"] = [0.85]
x_new["total sulfur dioxide"] = [120]
x_new_sc = scaler.transform(x_new)

# use the model to predict on the test data
model.predict(x_new_sc)
# -

# What if we want the predicted probabilities? For classification models, there is an additional method, `.predict_proba()`, that returns the predicted probability of each class.

model.predict_proba(x_new_sc)

model.classes_

# The first number represents the probability of the first class ("red") and the second number represents the probability of the second class ("white"). Notice that the predicted probabilities add up to 1, as they must.

# # Exercises

# **Exercise 1.** In the above example, we built a 9-nearest neighbors classifier to predict the color of a wine from just its volatile acidity and total sulfur dioxide. Use the model above to predict the color of a wine with the following features:
#
# - fixed acidity: 11
# - volatile acidity: 0.3
# - citric acid: 0.3
# - residual sugar: 2
# - chlorides: 0.08
# - free sulfur dioxide: 17
# - total sulfur dioxide: 60
# - density: 1.0
# - pH: 3.2
# - sulphates: 0.6
# - alcohol: 9.8
# - quality: 6
#
#
# Now, build a 15-nearest neighbors classifier using all of the features in the data set. Use this new model to predict the color of the same wine above.
#
# Does the predicted label change? Do the predicted probabilities of the labels change?

# +
# TYPE YOUR CODE HERE
# BEGIN SOLUTION
# %matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd
pd.options.display.max_rows = 5

reds = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/wines/reds.csv",
                   sep=";")
whites = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/wines/whites.csv", 
                     sep=";")

reds["color"] = "red"
whites["color"] = "white"

wines = pd.concat([reds, whites], 
                  ignore_index=True)
wines

# define the training data
X_train = wines[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
                "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]]
y_train = wines["color"]

X_train_sc = (X_train - X_train.mean()) / X_train.std()

x_new = pd.Series(index=X_train.columns)
x_new.loc[x_new.index] = [11,0.3,0.3,2,0.08,17,60,1.0,3.2,0.6,9.8,6]

x_new_sc = (x_new - X_train.mean()) / X_train.std()
display(x_new_sc)

dists = np.sqrt(((X_train_sc - x_new_sc) ** 2).sum(axis=1))
dists_sorted = dists.sort_values()
display(dists_sorted)

print("15 neighbors")
inds_nearest = dists_sorted.index[:15]
display(wines.loc[inds_nearest, "color"].value_counts())

# END SOLUTION
# -

# **Exercise 2.** Build a 5-nearest neighbors model to predict whether or not a passenger on a Titanic would survive, based on their age, sex, and class as features. Use the Titanic data set (`https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv`) as your training data. Then, use your model to predict whether a 20-year old female in first-class would survive. What about a 20-year old female in third-class?

# +
# TYPE YOUR CODE HERE
# BEGIN SOLUTION
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv')

# define the training data
cols = ["age","sex","pclass"]
X_train = df[cols]
y_train = df["survived"]


# convert categorical variables to dummy variables
vec = DictVectorizer(sparse=False)
X_train = X_train.to_dict(orient='records')
vec.fit(X_train)
X_train = pd.DataFrame(vec.transform(X_train))

X_train_sc = (X_train - X_train.mean()) / X_train.std()

x_new = pd.Series(index=cols)
x_new.loc[x_new.index] = [20,"female",1]

x_new = pd.DataFrame(vec.transform(x_new.to_dict()))

x_new_sc = (x_new - X_train.mean()) / X_train.std()
display(x_new_sc)

dists = np.sqrt(((X_train_sc - x_new_sc.loc[0]) ** 2).sum(axis=1))
dists_sorted = dists.sort_values()
display(dists_sorted)

print("5 neighbors")
inds_nearest = dists_sorted.index[:5]
display(df.loc[inds_nearest, "survived"].value_counts())
# END SOLUTION
# -
vec.get_feature_names()



