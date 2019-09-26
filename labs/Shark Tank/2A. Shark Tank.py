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

# # Shark Tank
#
# _Shark Tank_ is a reality TV show. Contestants present their idea for a company to a panel of investors (a.k.a. "sharks"), who then decide whether or not to invest in that company.  The investors give a certain amount of money in exchange for a percentage stake in the company ("equity"). If you are not familiar with the show, you may want to watch part of an episode [here](http://abc.go.com/shows/shark-tank) to get a sense of how it works.
#
# The data that you will examine in this lab contains data about all contestants from the first 6 seasons of the show, including:
# - the name and industry of the proposed company
# - whether or not it was funded (i.e., the "Deal" column)
# - which sharks chose to invest in the venture (N.B. There are 7 regular sharks, not including "Guest". Each shark has a column in the data set, labeled by their last name.)
# - if funded, the amount of money the sharks put in and the percentage equity they got in return
#
# To earn full credit on this lab, you should:
# - use built-in `pandas` methods (like `.sum()` and `.max()`) instead of writing a for loop over a `DataFrame` or `Series`
# - use the split-apply-combine pattern wherever possible
#
# Of course, if you can't think of a vectorized solution, a `for` loop is still better than no solution at all!

import pandas as pd

# ## Question 0. Getting and Cleaning the Data

# The data is stored in the CSV file `https://raw.githubusercontent.com/dlsun/data-science-book/master/data/sharktank.csv`. Read in the data into a Pandas `DataFrame`.

## YOUR CODE HERE
### BEGIN SOLUTION
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/sharktank.csv")
df.head()
### END SOLUTION

# There is one column for each of the sharks. A 1 indicates that they chose to invest in that company, while a missing value indicates that they did not choose to invest in that company. Notice that these missing values show up as NaNs when we read in the data. Fill in these missing values with zeros. Other columns may also contain NaNs; be careful not to fill those columns with zeros, or you may end up with strange results down the line.

## YOUR CODE HERE
### BEGIN SOLUTION
df.loc[:,["Corcoran","Cuban","Greiner","Herjavec","John","O'Leary","Harrington","Guest"]]=df.loc[:,["Corcoran","Cuban","Greiner","Herjavec","John","O'Leary","Harrington","Guest"]].fillna(0)
df.head()
### END SOLUTION

# Notice that Amount and Equity are currently being treated as categorical variables (`dtype: object`). Can you figure out why this is? Clean up these columns and cast them to numeric types (i.e., a `dtype` of `int` or `float`) because we'll need to perform mathematical operations on these columns.

## YOUR CODE HERE
### BEGIN SOLUTION
df["Amount"] = df["Amount"].fillna("$0").str[1:].str.replace(",","").astype(int)
df["Equity"] = df["Equity"].fillna("0%").str[:-1].astype(float)
df.head()
### END SOLUTION

# ## Question 1. Which Company was Worth the Most?

# The valuation of a company is how much it is worth. If someone invests \\$10,000 for a 40\% equity stake in the company, then this means the company must be valued at \$25,000, since 40% of \\$25,000 is \\$10,000.
#
# Calculate the valuation of each company that was funded. Which company was most valuable? Is it the same as the company that received the largest total investment from the sharks?

## YOUR CODE HERE
### BEGIN SOLUTION
df["Valuation"]=df["Amount"]/(df["Equity"]/100)
display(df.head())
display(df.set_index("Company")["Valuation"].idxmax())
display(df.set_index("Company")["Amount"].idxmax())
### END SOLUTION

# **ENTER YOUR WRITTEN EXPLANATION HERE.**
# ### BEGIN SOLUTION
# No. As you can see above, they are different.
# ### END SOLUTION

# ## Question 2. Which Shark Invested the Most?

# Calculate the total amount of money that each shark invested over the 6 seasons. Which shark invested the most total money over the 6 seasons? Avoid loops.
#
# _Hint:_ If $n$ sharks funded a given venture, then the amount that each shark invested is the total amount divided by $n$.

## YOUR CODE HERE
### BEGIN SOLUTION
df2 = df.copy()
df2.loc[:,["Corcoran","Cuban","Greiner","Herjavec","John","O'Leary","Harrington","Guest"]] = df.loc[:,["Company","Corcoran","Cuban","Greiner","Herjavec","John","O'Leary","Harrington","Guest"]].groupby("Company").apply(lambda x: x/sum(x.iloc[0,:])).fillna(0)
df2.loc[:,["Company","Amount","Corcoran","Cuban","Greiner","Herjavec","John","O'Leary","Harrington","Guest"]].groupby("Company").apply(lambda x: x.iloc[0,0]*x.iloc[0,1:]).sum().sort_values()
### END SOLUTION

# **ENTER YOUR WRITTEN EXPLANATION HERE.**
# ### BEGIN SOLUTION
# Cuban
# ### END SOLUTION

# ## Question 3. Do the Sharks Prefer Certain Industries?
#
# Calculate the funding rate (the proportion of companies that were funded) for each industry. Make a visualization showing this information.

## YOUR CODE HERE
### BEGIN SOLUTION
# %matplotlib inline
df.loc[:,["Industry","Amount"]].groupby("Industry").apply(lambda x: (sum(x.values > 0)/x.shape[0])[0]).sort_values().plot.bar()
### END SOLUTION

# **ENTER YOUR WRITTEN EXPLANATION HERE.**
# ### BEGIN SOLUTION
# Yes. Fitness and sports
# ### END SOLUTION
