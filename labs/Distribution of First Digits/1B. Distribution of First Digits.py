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

# # The Distribution of First Digits
#
# In this lab, you will explore the distribution of first digits in real data. For example, the first digits of the numbers 52, 30.8, and 0.07 are 5, 3, and 7 respectively. In this lab, you will investigate the question: how frequently does each digit 1-9 appear as the first digit of the number?

# ## Question 0
#
# Make a prediction. 
#
# 1. Approximately what percentage of the values do you think will have a _first_ digit of 1? What percentage of the values do you think will have a first digit of 9?
# 2. Approximately what percentage of the values do you think will have a _last_ digit of 1? What percentage of the values do you think will have a last digit of 9?
#
# (Don't worry about being wrong. You will earn full credit for any justified answer.)

# **ENTER YOUR WRITTEN EXPLANATION HERE.**

# ## Question 1
#
# The [S&P 500](https://en.wikipedia.org/wiki/S%26P_500_Index) is a stock index based on the market capitalizations of large companies that are publicly traded on the NYSE or NASDAQ. The CSV file `https://raw.githubusercontent.com/dlsun/data-science-book/master/data/sp500.csv` contains data from February 1, 2018 about the stocks that comprise the S&P 500. We will investigate the first digit distributions of the variables in this data set.
#
# Read in the S&P 500 data. What is the unit of observation in this data set? Is there a variable that is natural to use as the index? If so, set that variable to be the index. Once you are done, display the `DataFrame`.

## YOUR CODE HERE
### BEGIN SOLUTION
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/sp500.csv")
display(df.head())
display(df.tail())
print("Company on the stock exhange")
print("Yes. Name")
df.set_index("Name",inplace=True)
display(df.head())
### END SOLUTION

# **ENTER YOUR WRITTEN EXPLANATION HERE.**
# ### BEGIN SOLUTION
# Company on the stock exhange
#
# Yes. Name
# ### END SOLUTION

# ## Question 2
#
# We will start by looking at the `volume` column. This variable tells us how many shares were traded on that date.
#
# Extract the first digit of every value in this column. (_Hint:_ First, turn the numbers into strings. Then, use the [text processing functionalities](https://pandas.pydata.org/pandas-docs/stable/text.html) of `pandas` to extract the first character of each string.) Make an appropriate visualization to display the distribution of the first digits. (_Hint:_ Think carefully about whether the variable you are plotting is quantitative or categorical.)
#
# How does this compare with what you predicted in Question 0?

## YOUR CODE HERE
### BEGIN SOLUTION
# %matplotlib inline
df.volume.astype(str).str[0].value_counts().plot.bar()
### END SOLUTION

# **ENTER YOUR WRITTEN EXPLANATION HERE.**
# ### BEGIN SOLUTION
# Not know what I know, I would have assumed they were uniformly distributed. They are not uniform. 1 is most frequent.
# ### END SOLUTION

# ## Question 3
#
# Now, repeat Question 2, but for the distribution of _last_ digits. Again, make an appropriate visualization and compare with your prediction in Question 0.

## YOUR CODE HERE
### BEGIN SOLUTION
# %matplotlib inline
df.volume.astype(str).str[-1].value_counts().plot.bar()
### END SOLUTION

# **ENTER YOUR WRITTEN EXPLANATION HERE.**
# ### BEGIN SOLUTION
# This is closer to what I expect. Relatively uniform.
# ### END SOLUTION

# ## Question 4
#
# Maybe the `volume` column was just a fluke. Let's see if the first digit distribution holds up when we look at a very different variable: the closing price of the stock. Make a visualization of the first digit distribution of the closing price (the `close` column of the `DataFrame`). Comment on what you see.
#
# (_Hint:_ What type did `pandas` infer this variable as and why? You will have to first clean the values using the [text processing functionalities](https://pandas.pydata.org/pandas-docs/stable/text.html) of `pandas` and then convert this variable to a quantitative variable.)

## YOUR CODE HERE
### BEGIN SOLUTION
# %matplotlib inline
df.close.astype(str).str[1].value_counts().plot.bar()
### END SOLUTION

# **ENTER YOUR WRITTEN EXPLANATION HERE.**
# ### BEGIN SOLUTION
# Nope. 1 is still dominant.
# ### END SOLUTION


