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

# # 2.2 The Split-Apply-Combine Strategy

# %matplotlib inline
import pandas as pd

# In the previous section, we discussed how to restrict our analysis to a particular subset of observations using boolean masks. So, for example, if we wanted to calculate the survival rate for passengers in third class, we would write:

titanic_df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv")
titanic_df[titanic_df.pclass == 3].survived.mean()

# But what if we wanted to calculate the survival rate by class? We could slice the data set three times, once for each class:

(titanic_df[titanic_df.pclass == 1].survived.mean(), 
 titanic_df[titanic_df.pclass == 2].survived.mean(), 
 titanic_df[titanic_df.pclass == 3].survived.mean())

# But this code is inefficient and repetitive. It also does not generalize well to variables with hundreds of possible categories. 

# The problem of calculating the survival rate by class is an example of a problem that can be solved using the **split-apply-combine strategy**. The key insight here is that many data analyses follow the same basic pattern:
#
# - First, a data set is **split** into several subsets based on some variable.
# - Next, some analysis is **applied** to each subset.
# - Finally, the results from each analysis are **combined**.
#
# The three steps are diagrammed in the figure below:
#
# ![](split_apply_combine.png) [source](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.08-Aggregation-and-Grouping.ipynb)
#
# Applying this strategy to our working example above, we should first _split_ up the Titanic data according to the value of `pclass`, _apply_ `.survived.mean()` to each subset, and finally _combine_ the results into one `Series`.
#
# [_Note:_ The term "split-apply-combine" was coined by Hadley Wickham in [a 2011 paper](https://www.jstatsoft.org/article/view/v040i01), but the idea is not new. It should already be familiar to you if you know SQL or MapReduce.]

# ## Split-Apply-Combine in `pandas`: the `.groupby()` method
#
# To implement the split-apply-combine strategy in `pandas`, we use the `.groupby()` method. First, we specify one or more variables to split on in the argument to `.groupby()`. Then, we specify our analysis as usual. Pandas will handle splitting the data, applying the analysis to each subset, and combining the results at the end.

titanic_df.groupby("pclass").survived.mean()

# Compare this line of code with the code to calculate the overall survival rate:
#
# `titanic_df.survived.mean()`.
#
# The only difference is `.groupby("pclass")`. This turns a `DataFrame` into a `DataFrameGroupBy` object, which behaves like a `DataFrame`, except that any analysis that we specify will be applied to subsets of the `DataFrame` instead of the whole `DataFrame`.

# You can even make visualizations with `.groupby()`! To plot the age distribution of the survivors and non-survivors, we can group by the `survived` variable and then ask for a histogram of `age`. Behind the scenes, `pandas` will do this once for the survivors and again for the non-survivors and then combine them into one histogram.

titanic_df.groupby("survived").age.plot.hist(alpha=.5, normed=True, legend=True)

# It is also possible to group by more than one variable. Simply pass in a list of variable names to `.groupby()`. For example, the following code calculates the survival rate by class and sex:

survival_rates = titanic_df.groupby(["pclass", "sex"])["survived"].mean()
survival_rates

# It's clear that survival rates on the Titanic varied drastically by class and by sex.
#
# Notice that when we use `.groupby()`, the resulting index is whatever variable(s) we grouped by. Since we grouped by two variables, this index actually has two levels. An index with more than one level is called a `MultiIndex` in `pandas`. To access a particular row in a `DataFrame` that is indexed by a `MultiIndex`, we pass in a tuple of the values we want from each level.
#
# So, for example, to get female passengers in 2nd class, we would do:

survival_rates.loc[(2, "female")]

# If we pass in fewer values than there are levels in the index, `pandas` will return everything from the remaining levels.

survival_rates.loc[2]

# # Exercises
#
# Exercises 1-5 work with the Tips data set (`https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv`). The following code reads in the data into a `DataFrame` called `tips_df` and creates a new column called `tip_percent` out of the `tip` and `total_bill` columns. This new column represents the percentage tip paid (as a number between 0 and 1).

tips_df = pd.read_csv("/data301/data/tips.csv")
tips_df["tip_percent"] = tips_df.tip / tips_df.total_bill

# **Exercise 1.** On which day of the week does the waiter serve the largest parties, on average? (You did this exercise in the previous section. See how much easier it is to do using `.groupby()`.)

# +
# YOUR CODE HERE
# -

# **Exercise 2.** Calculate the average bill by day and time. What day-time combination has the highest average bill?

# +
# YOUR CODE HERE
# -

# **Exercise 3.** Extract the average bill for Friday lunch from the result of Exercise 2.

# +
# YOUR CODE HERE
# -

# **Exercise 4.** Use `.groupby()` to make a visualization comparing the distribution of tip percentages left by males and females. How do they compare?

# +
# YOUR CODE HERE
# -

# **Exercise 5.** Make a visualization that shows the average tip percentage as a function of table size.

# +
# YOUR CODE HERE
