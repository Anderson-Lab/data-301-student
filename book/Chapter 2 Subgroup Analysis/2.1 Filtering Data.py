# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md
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

# # Chapter 2. Subgroup Analysis
#
# # 2.1 Filtering Data

# +
# %matplotlib inline

import pandas as pd
pd.options.display.max_rows = 5
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
# -

# In the previous chapter, we only analyzed one variable at a time, but we always analyzed _all_ of the observations in a data set. But what if we want to analyze, say, only the passengers on the Titanic who were _male_? To do this, we have to **filter** the data. That is, we have to remove the rows of the `titanic_df` `DataFrame` where `sex` is not equal to `"male"`. In this section, we will learn several ways to obtain such a subsetted `DataFrame`.

# ## Two Ways to Filter a DataFrame
#
# One way to filter a `pandas` `DataFrame`, that uses a technique we learned in Chapter 1, is to set the filtering variable as the index and select the value you want using `.loc`.
#
# So for example, if we wanted a `DataFrame` with just the male passengers, we could do:

males = titanic_df.set_index("sex").loc["male"]
males

males.age.plot.hist()

# The more common way to filter a `DataFrame` is to use a **boolean mask**. A boolean mask is simply a `Series` of booleans whose index matches the index of the `DataFrame`.
#
# The easiest way to create a boolean mask is to use one of the standard comparison operators `==`, `<`, `>`, and `!=` on an existing column in the `DataFrame`. For example, the following code produces a boolean mask that is equal to `True` for the male passengers and `False` otherwise.

titanic_df.sex == "male"

# Notice that the equality operator `==` is not being used in the usual sense, i.e., to determine whether the object `titanic_df.sex` is the string `"male"`. This makes no sense, since `titanic_df.sex` is a `Series`. Instead, the equality operator is being _broadcast_ over the elements of `titanic_df.sex`. As a result, we end up with a `Series` of booleans that indicates whether _each_ element of `titanic_df.sex` is equal to `"male"`.
#
# This boolean mask can then be passed into a `DataFrame` to obtain just the subset of rows where the mask equals `True`.

titanic_df[titanic_df.sex == "male"]

# How can we tell that it worked? For one, notice that the index is missing the numbers 0 and 2; that's because passengers 0 and 2 in the original `DataFrame` were female. Also, the index goes up to 1308, but there are only 843 rows in this `DataFrame`. 
#
# In this new `DataFrame`, the variable `sex` should only take on one value, `"male"`. Let's check this.

titanic_df[titanic_df.sex == "male"]["sex"].value_counts()

# Now we can analyze this subsetted `DataFrame` using the techniques we learned in Chapter 1. For example, the following code produces a histogram of the ages of the male passengers on the Titanic:

titanic_df[titanic_df.sex == "male"].age.plot.hist()

# Boolean masks are also compatible with `.loc` and `.iloc`:

titanic_df.loc[titanic_df.sex == "male"]

# The ability to pass a boolean mask into `.loc` or `.iloc` is useful if we want to select columns at the same time that we are filtering rows. For example, the following code returns the ages of the male passengers:

titanic_df.loc[titanic_df.sex == "male", "age"]

# Of course, this result could be obtained another way; we could first apply the boolean mask and then select the column from the subsetted `DataFrame`, the same way we would select a column from any other `DataFrame`:

titanic_df[titanic_df.sex == "male"]["age"]

# ### Speed Comparison
#
# We've just seen two ways to filter a `DataFrame`. Which is better?
#
# One consideration is that the first method forces you to set the index of your `DataFrame` to the variable you want to filter on. If your `DataFrame` already has a natural index, you might not want to replace that index just to be able to filter the data.
#
# Another consideration is speed. Let's test the runtimes of the two options by using the `%timeit` magic. (**Warning:** The cell below will take a while to run, since `timeit` will run each command multiple times and report the mean and standard deviation of the runtimes.)

# %timeit titanic_df.set_index("sex").loc["male"].age.mean()
# %timeit titanic_df[titanic_df.sex == "male"].age.mean()

# So boolean masking is also significantly faster than re-indexing and selecting. All things considered, boolean masking is the best way to filter your data.

# ### Working with Boolean Series
#
# Remember that a boolean mask is a `Series` of booleans. A boolean variable is usually regarded as categorical, but it can also be regarded as quantitative, where `True`s are 1s and `False`s are 0s. For example, the following command actually produces a `Series` of 0s and 3s.

(titanic_df.sex == "male") * 3

# How can we use the dimorphic nature of booleans to our advantage? In Chapter 1.2, we saw how we functions like `.sum()` and `.mean()` could be applied to a binary categorical variable whose categories are coded as 0 and 1, such as the `survived` variable in the Titanic data set. The sum tells us the _number_ of observations in category 1, while the mean tells us the _proportion_ in category 1.
#
# Since boolean `Series` are essentially variables of 0s and 1s, the command

(titanic_df.sex == "male").sum()

# returns the _number_ of observations where `sex == "male"` and

(titanic_df.sex == "male").mean()

# returns the _proportion_ of observations where `sex == "male"`. Check that these answers are correct by some other method.

# ## Filtering on Multiple Criteria
#
# What if we want to visualize the age distribution of male _survivors_ on the Titanic?" To answer this question, we have to filter the `DataFrame` on two variables, `sex` and `survived`.
#
# We can filter on two or more criteria by combining boolean masks using logical operators. First, let's get the boolean masks for the two filters of interest:

titanic_df.sex == "male"

titanic_df.survived == 1

# Now, we want to combine these two boolean masks into a single mask that is `True` only when _both_ masks are `True`. This can be accomplished with the logical operator `&`.

(titanic_df.sex == "male") & (titanic_df.survived == 1)

# Verify for yourself that the `True` values in this `Series` correspond to observations where _both_ masks were True.

# _Warning:_ Notice the parentheses around each boolean mask above. These parentheses are necessary because of operator precedence. In Python, the logical operator `&` has higher precedence than the comparison operator `==`, so the command
#
# `titanic_df.sex == "male" & titanic_df.survived == 1`
#
# will be interpreted as 
#
# `titanic_df.sex == ("male" & titanic_df.survived) == 1`
#
# and result in an error. Python does not know how to evaluate `("male" & titanic_df.survived)`, since the logical operator `&` is not defined between a `str` and a `Series`. 
#
# The parentheses ensure that Python evaluates the boolean masks first and the logical operator second:
#
# `(titanic_df.sex == "male") & (titanic_df.survived == 1)`.
#
# It is very easy to forget these parentheses. Unfortunately, the error message that you get is not particularly helpful for debugging the code. If you don't believe me, just try running the offending command (without parentheses)!

# Now with the boolean mask in hand, we can plot the age distribution of male survivors on the Titanic:

titanic_df[(titanic_df.sex == "male") & (titanic_df.survived == 1)].age.plot.hist()

# Notice the peak between 0 and 10. A disproportionate number of young children survived because they were given priority to board the lifeboats.
#
# Besides `&`, there are two other logical operators, `|` and `~`, that can be used to modify and combine boolean masks.
#
# - `&` means "and"
# - `|` means "or"
# - `~` means "not"
#
# Like `&`, `|` and `~` operate elementwise on boolean `Series`. Examples are provided below.

# male OR survived
(titanic_df.sex == "male") | (titanic_df.survived == 1)

# equivalent to (titanic_df.sex != "male")
~(titanic_df.sex == "male")

# Notice how we use parentheses to ensure that the boolean mask is evaluated before the logical operators.

# # Exercises
#
# Exercises 1-3 deal with the Titanic data set.

# **Exercise 1.** Is there any advantage to selecting the column at the same time you apply the boolean mask? In other words, is the second option below any faster than the first?
#
# 1. `titanic_df[titanic_df.sex == "female"].age`
# 2. `titanic_df.loc[titanic_df.sex == "female", "age"]`
#
# Use the `%timeit` magic to compare the runtimes of these two options.

## YOUR CODE HERE
### BEGIN SOLUTION
# %timeit titanic_df[titanic_df.sex == "female"].age
# %timeit titanic_df.loc[titanic_df.sex == "female", "age"]
### END SOLUTION

# **Exercise 2.** Produce a graphic that compares the age distribution of the males who survived with the age distribution of the males who did not.

## YOUR CODE HERE
### BEGIN SOLUTION
titanic_df.loc[(titanic_df["sex"]=="male") & (titanic_df["survived"]==0)].age.plot.hist(alpha=0.5,legend=True,label="Survived=0")
titanic_df.loc[(titanic_df["sex"]=="male") & (titanic_df["survived"]==1)].age.plot.hist(alpha=0.5,legend=True,label="Survived=1")
### END SOLUTION

# **Exercise 3.** What proportion of 1st class passengers survived? What proportion of 3rd class passengers survived? See if you can use boolean masks to do this.

## YOUR CODE HERE
### BEGIN SOLUTION
display((titanic_df.loc[(titanic_df["pclass"]==1) & (titanic_df["survived"]==1)].shape[0])/titanic_df.loc[(titanic_df["pclass"]==1)].shape[0])
display((titanic_df.loc[(titanic_df["pclass"]==3) & (titanic_df["survived"]==1)].shape[0])/titanic_df.loc[(titanic_df["pclass"]==1)].shape[0])
### END SOLUTION

# Exercises 4-7 ask you to analyze the Tips data set (`https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv`). The following code reads the data into a `DataFrame` called `tips_df` and creates a new column called `tip_percent` out of the `tip` and `total_bill` columns. This new column represents the tip as a percentage of the total bill (as a number between 0 and 1).

tips_df = pd.read_csv("https://raw.githubusercontent.com/dlsun/data-science-book/master/data/tips.csv")
tips_df["tip_percent"] = tips_df.tip / tips_df.total_bill
tips_df

# **Exercise 4.** Calculate the average tip percentage paid by parties of 4 or more.

## YOUR CODE HERE
### BEGIN SOLUTION
tips_df.loc[tips_df["size"]>=4,:].tip_percent.mean()
### END SOLUTION

# **Exercise 5.** Make a visualization comparing the distribution of tip percentages left by males and females. How do they compare?

## YOUR CODE HERE
### BEGIN SOLUTION
tips_df.loc[tips_df["sex"]=="Male",:].tip_percent.plot.hist(legend=True,alpha=0.5,density=True,label="Male")
tips_df.loc[tips_df["sex"]=="Female",:].tip_percent.plot.hist(legend=True,alpha=0.5,density=True,label="Female")
### END SOLUTION

# **Exercise 6.** What is the average table size on weekdays? (_Hint:_ There are at least two ways to create the appropriate boolean mask: using the `|` logical operator and using the `.isin()` method. See if you can do it both ways.)

## YOUR CODE HERE
### BEGIN SOLUTION
print(tips_df.loc[tips_df["day"].isin(["Mon","Tue","Wed","Thur","Fri"]),:]["size"].mean())
print(tips_df.loc[(tips_df["day"] == "Mon") | (tips_df["day"] == "Tue") | (tips_df["day"] == "Wed") | (tips_df["day"] == "Thur") | (tips_df["day"] == "Fri"),:]["size"].mean())
### END SOLUTION

# **Exercise 7.** Calculate the average table size for each day of the week. On which day of the week does the waiter serve the largest parties, on average?

## YOUR CODE HERE
### BEGIN SOLUTION
print("Mon",tips_df.loc[(tips_df["day"] == "Mon"),"size"].mean())
print("Tues",tips_df.loc[(tips_df["day"] == "Tues"),"size"].mean())
print("Wed",tips_df.loc[(tips_df["day"] == "Wed"),"size"].mean())
print("Thur",tips_df.loc[(tips_df["day"] == "Thur"),"size"].mean())
print("Fri",tips_df.loc[(tips_df["day"] == "Fri"),"size"].mean())
print("Sat",tips_df.loc[(tips_df["day"] == "Sat"),"size"].mean())
print("Sun",tips_df.loc[(tips_df["day"] == "Sun"),"size"].mean())
### END SOLUTION


