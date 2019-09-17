# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Chapter 4.2 Distance Metrics and Categorical Variables
#
#

# The distance metrics that we studied in the previous section were designed for quantitative variables. But most data sets contain a mix of categorical and quantitative variables. For example, the Titanic data set contains both quantitative variables, like `age`, and categorical variables, like `sex` and `embarked`. How do we measure the similarity between observations for a data set like this one? The most straightforward solution is to convert the categorical variables into quantitative ones.

# +
# %matplotlib inline
import numpy as np
import pandas as pd
pd.options.display.max_rows = 5

titanic = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv")
titanic
# -

# ## Converting Categorical Variables to Quantitative Variables
#
# Binary categorical variables (categorical variables with two categories) can be converted into quantitative variables by coding one category as 1 and the other category as 0. (In fact, the `survived` column in the Titanic data set is an example of a variable where this has been done.) But what do we do about a categorical variable with more than 2 categories, like `embarked`, which has 3 categories?
#
# We can convert a categorical variable with $K$ categories into $K$ separate 0/1 variables, or **dummy variables**. Each of the $K$ variables is an indicator for one of the $K$ categories. That is, each dummy variable is 1 if the observation fell into that category and 0 otherwise.
#
# Although it is not difficult to create dummy variables manually, the easiest way to create them is the `get_dummies()` function in `pandas`.

pd.get_dummies(titanic["embarked"])

# Since every observation is in exactly one category, each row contains exactly one 1; the rest of the values in each row are 0s.
#
# We can call `get_dummies` on a `DataFrame` to encode multiple categorical variables at once. `pandas` will only dummy-encode the variables it deems categorical, leaving the quantitative variables alone. If there are any categorical variables that are represented in the `DataFrame` using numeric types, they must be cast explicitly to a categorical type, such as `str`.  `pandas` will also automatically prepend the variable name to all dummy variables, to prevent collisions between column names in the final `DataFrame`.

# +
# Convert pclass to a categorical type
titanic["pclass"] = titanic["pclass"].astype(str)

# Pass all variables to get_dummies, except ones that are "other" types
titanic_num = pd.get_dummies(
    titanic.drop(["name", "ticket", "cabin", "boat", "body"], axis=1)
)
titanic_num
# -

# Notice that categorical variables, like `pclass`, were converted to dummy variables with names like `pclass_1`, `pclass_2` and `pclass_3`, while quantitative variables, like `age`, were left alone.

# Now that we have converted every variable in our data set into a quantitative variable, we can apply the techniques from the previous section (Section 4.1) to calculate distances between observations. For example, to find the passenger who is most similar to the first passenger, Elisabeth Watson, we can find the row with the smallest Euclidean distance to that row in the above `DataFrame`.

titanic_std = (titanic_num - titanic_num.mean()) / titanic_num.std()
np.sqrt(
    ((titanic_std - titanic_std.loc[0]) ** 2).sum(axis=1)
).sort_values()

# The passenger who was most similar to Elisabeth Allen, other than herself, is passenger 238. Let's extract these passengers from the original `DataFrame` to see how similar they really are.

titanic.loc[[0, 238]]

# The two passengers are indeed very similar, only differing in age and the number of parents/children accompanying her. They even happen to share the same first two names ("Elizabeth Walton").

# # Exercises
#
# Exercises 1 and 2 use the Ames housing data set (`https://raw.githubusercontent.com/dlsun/data-science-book/master/data/AmesHousing.txt`).

# **Exercise 1.** The neighborhood variable (`Neighborhood`) in this data set is categorical. Convert it to $K$ quantitative variables. What is $K$ in this case?
#
# Based on these $K$ variables only, calculate the Euclidean distance between house 0 and each of the other houses in the data set. What are the possible values of the Euclidean distance? Can you explain what a distance of $0$ means, in the context of this variable? What about a distance of $1$?

# +
# ENTER YOUR CODE HERE
# -

# **Exercise 2.** Suppose that you really like house 0 in the data set, but it is too expensive. Find cheaper homes that are similar to it, by calculating distances after encoding categorical variables as dummy variables. Be sure to actually look at the profiles of the homes that your algorithm picked out as most similar. Do they make sense?
#
# Try different distance metrics and different standardization methods. How sensitive are your results to these choices?
#
# _Think:_ If the goal is to find a "good deal" on a similar house, should sale price be included as a variable in your distance metric? 
#
# _Hint:_ There are too many variables in the data set. Do not try to call `pd.get_dummies()` on the entire `DataFrame`! You will want to pare down the number of variables, but be sure to include a mixture of categorical and quantitative variables. Refer to the [data documentation](https://ww2.amstat.org/publications/jse/v19n3/decock/DataDocumentation.txt) for information about the variables.

# +
# ENTER YOUR CODE HERE
