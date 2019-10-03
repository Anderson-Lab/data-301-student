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

# # Chapter 3. Relationships between Variables
#
# So far, we have seen different ways to summarize and visualize _individual_ variables in a data set. But we have not really discussed how to summarize and visualize relationships between _multiple_ variables. This chapter is all about how to understand relationships between the columns in a `DataFrame`. The methods will be different, depending on whether the variables are categorical or quantitative.

# # 3.1 Relationships between Categorical Variables
#
# In this section, we look at ways to summarize the relationship between two _categorical_ variables. To do this, we will again use the Titanic data set.

# +
# %matplotlib inline
import pandas as pd

titanic_df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv")
# -

# Suppose we want to understand the relationship between where a passenger embarked and what class they were in. We can completely summarize this relationship by counting the number of passengers in each class that embarked at each location. We can create a pivot table that summarizes this information.

embarked_pclass_counts = titanic_df.pivot_table(
    index="embarked", columns=["pclass"],
    values="name",  # We can pretty much count any column, as long as there are no NaNs.
    aggfunc="count" # The count function will count the number of non-null values.
)
embarked_pclass_counts

# A pivot table that stores counts is also called a **contigency table** or a **cross-tabulation**. This type of pivot table is common enough that there is a specific function in `pandas` to calculate one, allowing you to bypass `.pivot_table`:

titanic_df.shape

counts = pd.crosstab(titanic_df.embarked, titanic_df.pclass)
counts.sum(axis=0)/counts.sum().sum()

# ## Joint Distributions

# It is common to normalize the counts in a table so that they add up to 1. These proportions represent the **joint distribution** of the two variables.
#
# To calculate the joint distribution, we need to divide the table of counts above by the total count. To find the total count, we call `.sum()` twice; the first call gives us the sum of each column, and the second call adds those numbers together.

print(counts.sum().sum())
joint = counts / counts.sum().sum()
joint

# Note that this is yet another example of broadcasting. When we divided the `DataFrame` `counts` by the number 1307, the division was applied elementwise, producing another `DataFrame`.
#
# Each cell in this `DataFrame` tells us a joint proportion. For example, the cell in the bottom right tells us the proportion of all passengers that embarked at Southampton and were in 3rd class. We notate this joint proportion as follows:
#
# $$ P(\text{embarked at Southampton and in 3rd class}) = .379. $$
#
# The joint distribution above could also have been obtained by specifying `normalize=True` when the contingency table was first created:

pd.crosstab(titanic_df.embarked, titanic_df.pclass,
            normalize=True)

# The above joint distribution is not, strictly speaking, a contingency table. A contingency table is a table of all counts, while the above table is a table of proportions.

# ## Marginal Distributions
#
# The **marginal distribution** of a variable is simply the distribution of that variable, ignoring the other variables. To calculate the marginal distribution from a joint distribution of two variables, we sum the rows or the columns of the joint distribution.
#
# For example, to calculate the marginal distribution of `embarked`, we have to sum the joint distribution over the columns---in other words, _roll-up_ or _marginalize over_ the `pclass` variable:

joint.sum(axis=1)

# We can check this answer by calculating the distribution of `embarked` directly from the original data, ignoring `pclass` entirely.

embarked_counts = titanic_df.groupby("embarked")["name"].count()
embarked_counts / embarked_counts.sum()

# The numbers match!
#
# Likewise, we calculate the marginal distribution of `pclass` by summing the joint distribution over the rows---in other words, by _rolling-up_ or _marginalizing over_ the `embarked` variable:

joint.sum(axis=0)

# So given the joint distribution of two categorical variables, there are two marginal distributions: one for each of the variables. These marginal distributions are obtained by summing the joint distribution table over the rows and over the columns.
#
# The _marginal distribution_ is so-named because these row and column totals would typically be included alongside the joint distribution, in the _margins_ of the table. A contingency table with the marginal distributions included can be obtained by specifying `margins=True` in `pd.crosstab`:

pd.crosstab(titanic_df.embarked, titanic_df.pclass,
            normalize=True, margins=True)

# ## Conditional Distributions
#
# The **conditional distribution** tells us about the distribution of one variable, _conditional on_ the value of another. For example, we might want to know the proportion of 3rd class passengers that embarked at each location. In other words, what is the distribution of where a passenger embarked, _conditional on_ being in 3rd class?
#
# If we go back to the contingency table:

embarked_pclass_counts

# there were $101 + 113 + 495 = 709$ passengers in 3rd class, of whom 
#
# - $101 / 709 = .142$ were in 1st class,
# - $113 / 709 = .159$ were in 2nd class, and
# - $495 / 709 = .698$ were in 3rd class.
#
# We can calculate these proportions in code by dividing the `pclass=3` column by its sum:

embarked_pclass_counts[3] / embarked_pclass_counts[3].sum()

# Notice that these three proportions add up to 1, making this a proper distribution.
#
# This conditional distribution helps us answer questions such as, "What proportion of 3rd class passengers embarked at Southampton?" We notate this conditional proportion as follows:
#
# $$ P\big(\textrm{embarked at Southampton}\ \big|\ \textrm{in 3rd class}\big) = 0.698. $$
#
# The pipe $\big|$ is read "given". So we are interested in the proportion of passengers who embarked at Southampton, _given_ that they were in 3rd class.
#
# We could have also calculated this conditional distribution from the joint distribution (i.e., proportions instead of counts):

joint[3] / joint[3].sum()

# We have just calculated _one_ of the conditional distributions of `embarked`: the distribution conditional on being in 3rd class. There are two more conditional distributions of `embarked`: 
#
# - the distribution conditional on being in 1st class 
# - the distribution conditional on being in 2nd class
#
# It is common to report _all_ of the conditional distributions of one variable given another variable.
#
# Of course, it is straightforward to calculate these conditional distributions manually:

embarked_pclass_counts[1] / embarked_pclass_counts[1].sum()

embarked_pclass_counts[2] / embarked_pclass_counts[2].sum()

# But there is a nifty trick for calculating all three conditional distributions at once. By summing the counts over `embarked`, we obtain the total number of people in each `pclass`:

pclass_counts = embarked_pclass_counts.sum(axis=0)
pclass_counts

# This is exactly what we need to divide each column of `embarked_pclass_counts` by:

embarked_given_pclass = embarked_pclass_counts.divide(pclass_counts, axis=1)
embarked_given_pclass

# (This is yet another example of _broadcasting_, since we are dividing a `DataFrame` by a `Series`.)
#
# Compare each column with the numbers we obtained earlier. Notice also that each column sums to 1, a reminder that each column represents a separate distribution.
#
# When comparing numbers across distributions, it is important to be careful. For example, the 87.4% and the 69.8% in the "Southampton" row represent percentages of different populations. Just because 87.4% is higher than 69.8% does not mean that more 2nd class passengers boarded at Southampton than 3rd class passengers. In fact, if we go back to the original contingency table, we see that more 3rd class passengers actually boarded at Southampton than 2nd class passengers!

# There is also another set of conditional distributions for these two variables: the distribution of class, conditional on where they embarked. To calculate these conditional distributions, we instead divide `embarked_pclass_counts` by the sum of each row:

embarked_counts = embarked_pclass_counts.sum(axis=1)
display(embarked_counts)
display(embarked_pclass_counts)
pclass_given_embarked = embarked_pclass_counts.divide(embarked_counts, axis=0)
pclass_given_embarked

# These conditional distributions answer questions like, "What proportion of Southampton passengers were in 3rd class?" 
#
# Notice that these proportions are _not_ the same as the proportions for the other set of conditional distributions. That is because the two questions below are fundamentally different:
#
# _Question 1._ What proportion of 3rd class passengers embarked at Southampton?
#
# $$P\big(\textrm{embarked at Southampton}\ \big|\ \textrm{in 3rd class}\big) = \frac{\text{passengers who embarked at Southampton and in 3rd class}}{\text{ passengers who in 3rd class}}$$
#
# _Question 2._ What proportion of Southampton passengers were in 3rd class? 
#
# $$P\big(\textrm{in 3rd class}\ \big|\ \textrm{embarked at Southampton}\big) = \frac{\text{passengers who embarked at Southampton and in 3rd class}}{\text{passengers who embarked at Southampton}} \\ $$
#
#
#
# In the first case, the reference population is all passengers who embarked at Southampton. In the second case, the reference population is all passengers who were in 3rd class. The numerators may be the same, but the denominators are different. In general, the conditional distributions of $X$ given $Y$ are _not_ the same as the conditional distributions of $Y$ given $X$. 
#
# If we rephrase the question slightly, we get yet another answer:
#
# _Question 3._ What proportion of passengers embarked at Southampton _and_ were in 3rd class?
#
# $$P(\text{embarked at Southampton and in 3rd class}) = \frac{\text{passengers who embarked at Southampton and in 3rd class}}{\text{passengers (total)}}$$
#
# The reference population here is all passengers. This is the proportion that one would get from the joint distribution.
#
# It is important to pay attention to the wording of the question, to determine whether a joint distribution or a conditional distribution is called for---and, if the latter, which of the two conditional distributions is appropriate.

# ## Visualization
#
# How do we visualize the joint and conditional distributions of two categorical variables? (Marginal distributions are summaries of a single variable and can be visualized using the techniques of Chapter 1.)
#
# To visualize a joint distribution, we need to be able to represent three dimensions: two dimensions for the two categorical variables and a third dimension for the proportions. Although one option is a 3D graph, humans are not good at judging the sizes of 3D objects printed on a page. For this reason, **heat maps**, which use a color scale to represent the third dimension, are usually preferred. 
#
# Unfortunately, heat maps are still not easy to create in `pandas`. We use the `seaborn` library to make a heat map:

# +
import seaborn as sns

sns.heatmap(joint)
# -

# A heat map encourages comparison across cells. So we see that 3rd class passengers who embarked at Southampton were by far the most common.
#
# Although a heat map can also be used to visualize conditional distributions, it is not ideal because it does not tell us which variable we are conditioning on, and it is difficult to judge visually which dimension sums to 1. A stacked bar graph is better because it visually shows values summing to 1.
#
# To make a stacked bar graph, we simply specify `stacked=True` in `.plot.bar()`, to get the bars to show up on top of one another, instead of side-by-side:

pclass_given_embarked.plot.barh(stacked=True)

# However, the same code does not work on the other set of conditional distributions:

embarked_given_pclass.plot.bar(stacked=True)

# What went wrong? Recall that `.plot.bar()` automatically plots the (row) index of the `DataFrame` on the $x$-axis. To plot the distribution of `embarked` conditional on `pclass`, we need `pclass` to be on the $x$-axis, but 

embarked_given_pclass

# has `embarked` as the index. To make `pclass` the index, we can **transpose** this `DataFrame` so that the rows become columns and the columns become rows. The syntax for transposing a `DataFrame` is `.T`, which is inspired by the notation for transposing a matrix in linear algebra.

embarked_given_pclass.T

# Now, we can make a stacked bar graph from this _transposed_ `DataFrame`:

(embarked_given_pclass.T).plot.bar(stacked=True)

# # Exercises

# Exercises 1-4 deal with the Tips data set (`https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv`).

# **Exercise 1.** Make a visualization that displays the relationship between the day of the week and party size.

# YOUR CODE HERE
# BEGIN SOLUTION
df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv")
size_day = pd.crosstab(df["size"], df["day"])
display(size_day)
display(size_day.sum(axis=0))
display(size_day.divide(size_day.sum(axis=0),axis=1))
display(size_day.divide(size_day.sum(axis=0),axis=1).transpose().plot.bar(stacked=True))
# END SOLUTION

# **Exercise 2.** Calculate the marginal distribution of day of week in two different ways.

# ENTER YOUR CODE HERE
# BEGIN SOLUTION
joint = pd.crosstab(df["day"], df["size"],
            normalize=True, margins=True)
display(joint)
(pd.crosstab(df["day"], df["size"])/df.shape[0]).sum(axis=1)
# END SOLUTION

# **Exercise 3.** Make a visualization that displays the conditional distribution of party size, given the day of the week.

# ENTER YOUR CODE HERE
# BEGIN SOLUTION
joint = pd.crosstab(df["day"], df["size"])
display(joint)
display(joint.sum(axis=1))
display(joint.divide(joint.sum(axis=1),axis=0))
# END SOLUTION

# **Exercise 4.** What proportion of Saturday parties had 2 people? Is this the same as the proportion of 2-person parties that dined on Saturday?

# ENTER YOUR CODE HERE
# BEGIN SOLUTION
joint = pd.crosstab(df["day"], df["size"])
display(joint)
display(joint.sum(axis=1))
display(joint.divide(joint.sum(axis=1),axis=0))
# END SOLUTION

# **Challenge Exercise.** We discussed above that the conditional distributions of A given B and the conditional distributions of B given A are _not_ the same. Can you figure out a way to relate the two? Can you write code that will convert a table with the conditional distributions of A given B, into a table with the conditional distributions of B given A?

# +
# ENTER YOUR CODE HERE
# BEGIN SOLUTION
# END SOLUTION
