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

# # 3.3 Relationships between Two Quantitative Variables
#
# In this chapter, we discuss ways to summarize and visualize relationships between _quantitative_ variables. To illustrate the concepts, we use the Ames housing data set.

# +
# %matplotlib inline
import pandas as pd

housing_df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/AmesHousing.txt", sep="\t")
housing_df.head()
# -

# ## Visualization
#
# Let's start by visualizing the relationship between the square footage (of the dwelling) and the sale price, both of which are quantitative variables. To do this, we can make a **scatterplot**. In a scatterplot, each observation is represented by a point. The $(x, y)$ coordinates of each point represent the values of two variables for that observation.
#
# To make a scatterplot in `pandas`, we use the `.plot.scatter()` method of `DataFrame`. Since there are multiple columns in the `DataFrame`, we have to specify which variable is $x$ and which variable is $y$.

housing_df.plot.scatter(x="Gr Liv Area", y="SalePrice")

# We see that square footage (of the dwelling) and the sale price have a positive relationship. That is, the greater the living area, the higher the sale price.

# ## Summary Statistics
#
# To summarize the relationship between two quantitative variables $X$ and $Y$, we can report the _covariance_ between them, defined as 
#
# $$ \text{Cov}[X, Y] = \frac{1}{n - 1} \sum_x \sum_y (x - \text{mean}[X]) (y - \text{mean}[Y])$$
#
# The sign of the covariance will match the direction of the relationship between the two variables. The figures below illustrate why. If two variables are positively related, then the scatterplot might look as follows, with most points in the upper-right and lower-left quadrants (when you divide up the plane into four quadrants based on the means of $X$ and $Y$).
#
# ![](positive_cov.png)
#
# Each point on this scatterplot contributes to the sum that makes up the covariance. Any point in the upper-right quadrant (where $x$ and $y$ are both greater than their respective means) has a positive contribution, since the product of two positive numbers is positive. A point in the lower-left quadrant (where $x$ and $y$ are both less than their respective means) also has a positive contribution, since the product of two negative numbers is also positive. Therefore, on the whole, the covariance will be positive for two variables with a positive relationship.
#
# We can also consider two variables with a negative relationship. A scatterplot of two negatively-related variables might look as follows, with most points in the upper-left and lower-right quadrants. Points in both of these quadrants will have a negative contribution towards the covariance, since the product of a positive and a negative number is negative.
#
# ![](negative_cov.png)
#
# What does it mean for the covariance to be _zero_? It does not necessarily mean that there is _no_ relationship at all between the two variables; it just means that the two variables do not move in a consistent direction. For example, the two variables below have _zero_ covariance because the negative contributions from the upper-left and lower-right quadrants perfectly cancel out the positive contributions from the upper-right and lower-left quadrants. However, it would be inaccurate to say that $X$ and $Y$ have _no_ relationship; they have a strong relationship, but it just is not consistently in one direction.
#
# ![](zero_cov.png)
#
# To calculate the covariance between two quantitative variables, we use the `.cov()` method in `pandas`. This method is attached to one `Series` and takes another `Series` of the same length as input. It returns the covariance between the two `Series`.

housing_df["Gr Liv Area"].cov(housing_df["SalePrice"])

# The covariance between the two variables is positive, as should be apparent from the scatterplot above. Larger houses sell for higher prices.
#
# One criticism of the covariance is that the value itself is difficult to interpret, and covariances are not comparable across different variables.  As we did with the $\chi^2$ distance in the previous section, we can normalize the covariance. This _normalized covariance_ is called the **correlation** and is symbolized $r$:
#
# $$ r = \frac{\text{Cov}[X, Y]}{\text{SD}[X] \text{SD}[Y]} $$
#
# The correlation has all of the important properties of covariance: 
#
# - A positive correlation indicates a positive relationship between the variables. As one increases, so does the other.
# - A negative correlation indicates a negative relationship between the variables. As one increases, the other tends to decrease.
# - A zero correlation means that the two variables do not move in a consistent direction, but does not necessarily mean that they have _no_ relationship.
#
# But the correlation is also guaranteed to be between $-1$ and $1$, so it can be compared across data sets.
#
# What does a maximal correlation of $\pm 1$ mean? It means that the data fall perfectly along a line.
#
# ![](corr_1.png)
#
# Correlation is calculated in `pandas` in much the same way that covariance is, using the `.corr()` method:

housing_df["Gr Liv Area"].corr(housing_df["SalePrice"])

# Like the covariance, the correlation $r$ is positive, but it is a number between $-1$ and $+1$. $r=+1$ would mean that all of the points on the scatterplot fell perfectly along a line (with positive slope). Although the points in the scatterplot do not all fall perfectly on a line, they do seem to hover around an underlying line. This explains why the covariance is close to, but not equal to, $1$.

# # Exercises

# **Exercise 1.** What is the correlation between any variable and itself? Check your answer with any (quantitative) variable from the Ames housing data set.

# ENTER YOUR CODE HERE
# BEGIN SOLUTION
housing_df["Gr Liv Area"].corr(housing_df)
# END SOLUTION

# Exercises 2-3 deal with the Tips data set (`https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv`).

# **Exercise 2.** Make a scatterplot showing the relationship between the tip and the total bill.

# ENTER YOUR CODE HERE
# BEGIN SOLUTION
tips = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv")
tips.plot.scatter(x="total_bill",y="tip")
# END SOLUTION

# **Exercise 3.** Which pair of variables in this data set have the highest correlation with each other?

# ENTER YOUR CODE HERE
# BEGIN SOLUTION
display(tips.corr())
print("Total bill and tip")
# END SOLUTION

# **Exercise 4.** To build your intuition about correlation, play this [correlation guessing game](http://guessthecorrelation.com/). There is even a two-player mode that allows you to play against a friend in the class.
