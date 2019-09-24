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

# # Chapter 1. Tables, Observations, and Variables
#
# # 1.1 Introduction to Tabular Data
#
# What does data look like? For most people, the first image that comes to mind is a spreadsheet, with numbers neatly arranged in a table of rows and columns. One goal of this book is to get you to think beyond tables of numbers---to recognize that the words in a book and the markers on a map are also data to be collected, processed, and analyzed. But a lot of data is still organized into tables, so it is important to know how to work with **tabular data**. In fact, most machine learning algorithms can only process data that can be roughly considered **tabular data**. Where does this show up?

# Let's look at a tabular data set. Shown below are the first 5 rows of a data set about the passengers on the Titanic. This data set contains information about each passenger (e.g., name, sex, age), their journey (e.g., the fare they paid, their destination), and their ultimate fate (e.g., whether they survived or not, the lifeboat they were on).
#
# <img src="titanic_data.png" width="800">

# In a tabular data set, each row represents a distinct observation and each column a distinct variable. Each **observation** is an entity being measured, and **variables** are the attributes we measure. In the Titanic data set above, each row represents a passenger on the Titanic. For each passenger, 14 variables have been recorded, including `pclass` (their ticket class: 1, 2, or 3) and `boat` (which lifeboat they were on, if they survived).

# ## Storing Data on Disk and in Memory

# How do we represent tabular data on disk so that it can be saved for later or shared with someone else? The Titanic data set above is saved in a file called `titanic.csv`. Let's peek inside this file using the shell command `head`.
#
# _Jupyter Tip_: To run a shell command inside a Jupyter notebook, simply prefix the shell command by the `!` character.
#
# _Jupyter Tip_: To run a cell, click on it and press the "play" button in the toolbar above. (Alternatively, you can press `Shift+Enter` on the keyboard.)

# !head /data301/data/titanic.csv

# The first line of this file contains the names of the variables, separated by commas. Each subsequent line contains the values of those variables for a passenger.  The values appear in the same order as the variable names in the first line and are also separated by commas. Because the values in this file are separated (or _delimited_) by commas, this file is called a **comma-separated values** file, or **CSV** for short. CSV files typically have a `.csv` file extension, but not always.
#
# Although commas are by far the most common delimiter, you may encounter tabular data files that use tabs, semicolons (;), or pipes (|) as delimiters.

# How do we represent this information in memory so that it can be manipulated efficiently? In Python, the `pandas` library provides a convenient data structure for storing tabular data, called the `DataFrame`.

import pandas as pd
pd.DataFrame

# To read a file from disk into a `pandas` `DataFrame`, we can use the `read_csv` function in `pandas`. The first line of code below reads the Titanic dataset into a `DataFrame` called `df`. The second line calls the `.head()` method of `DataFrame`, which returns a new `DataFrame` consisting of just the first few rows (or "head") of the original.

df = pd.read_csv("/data301/data/titanic.csv")
df.head()

# _Jupyter Tip_: When you execute a cell in a Jupyter notebook, the result of the last line is automatically printed. To suppress this output, you can do one of two things:
#
# - Assign the result to a variable, e.g., `df_head = df.head()`.
# - Add a semicolon to the end of the line, e.g., `df.head();`.
#
# I encourage you to try these out by modifying the code above and re-running the cell!

# Now that the tabular data is in memory as a `DataFrame`, we can manipulate it by writing Python code.

# ## Observations
#
# Recall that **observations** are the rows in a tabular data set. It is important to think about what each row represents, or the **unit of observation**, before starting a data analysis. In the Titanic `DataFrame`, the unit of observation is a passenger. This makes it easy to answer questions about passengers (e.g., "What percentage of passengers survived?") but harder to answer questions about families (e.g., "What percentage of families had at least one surviving member?")

# What if we instead had one row per _family_, instead of one row per _passenger_? We could still store information about _how many_ members of each family survived, but this representation would make it difficult to store information about _which_ members survived.
#
# There is no single "best" representation of the data. The right representation depends on the question you are trying to answer: if you are studying families on the Titanic, then you might want the unit of observation to be a family, but if you need to know which passengers survived, then you might prefer that it be a passenger. No matter which representation you choose, it is important to be conscious of the unit of observation.

# ### The Row Index
#
# In a `DataFrame`, each observation is identified by an index. You can determine the index of a `DataFrame` by looking for the **bolded** values at the beginning of each row when you print the `DataFrame`. For example, notice how the numbers **0**, **1**, **2**, **3**, **4**, ... above are bolded, which means that this `DataFrame` is indexed by integers starting from 0. This is the default index when you read in a data set from disk into `pandas`, unless you explicitly specify otherwise.

# Since each row represents one passenger, it might be useful to re-index the rows by the name of the passenger. To do this, we call the `.set_index()` method of `DataFrame`, passing in the name of the column we want to use as the index. Notice how `name` now appears at the very left, and the passengers' names are all bolded. This is how you know that `name` is the index of this `DataFrame`.

df.set_index("name").head()

# _Warning_: The `.set_index()` method does _not_ modify the original `DataFrame`. It returns a _new_ `DataFrame` with the specified index. To verify this, let's look at `df` again after running the above code.

df.head()

# Nothing has changed! If you want to save the `DataFrame` with the new index, you have to explicitly assign it to a variable.

df_by_name = df.set_index("name")
df_by_name.head()

# If you do not want the modified `DataFrame` to be stored in a new variable, you can either assign the result back to itself:
#
# `df = df.set_index("name")`
#
# or use the `inplace=True` argument, which will modify the `DataFrame` in place:
#
# `df.set_index("name", inplace=True)`.
#
# These two commands should only be run once. If you try to run them a second time, you will get an error. Don't just take my word for it---create a cell below and try it! The reason for the error is: after the command is executed the first time, `name` is no longer a column in `df`, since it is now in the index. When the command is run again, `pandas` will try (and fail) to find a column called `name`. 
#
# Thus, the interactivity of Jupyter notebooks is both a blessing and a curse. It allows us to see the results of our code immediately, but it makes it easy to lose track of the state, especially if you run a cell twice or out of order. Remember that Jupyter notebooks are designed to be run from beginning to end. Keep this in mind as you run other people's notebooks and as you organize your own notebooks.

# ### Selecting Rows
#
# Now that we have set the (row) index of the `DataFrame` to be the passengers' names, we can use the index to select specific passengers. To do this, we use the `.loc` selector. The `.loc` selector takes in a label and returns the row(s) corresponding to that index label.
#
# For example, if we wanted to find the data for the father of the Allison family, we would pass in the label "Allison, Master. Hudson Trevor" to `.loc`. Notice the square brackets. 

df_by_name.loc["Allison, Master. Hudson Trevor"]

# Notice that the data for a single row is printed differently. This is no accident. If we inspect the type of this data structure:

type(df_by_name.loc["Allison, Master. Hudson Trevor"])

# we see that it is not a `DataFrame`, but a different data structure called a `Series`.

# `.loc` also accepts a _list_ of labels, in which case it returns multiple rows, one row for each label in the list. So, for example, if we wanted to select all 4 members of the Allison family from `df_by_name`, we would pass in a list with each of their names.

df_by_name.loc[[
    "Allison, Master. Hudson Trevor",
    "Allison, Miss. Helen Loraine",
    "Allison, Mr. Hudson Joshua Creighton",
    "Allison, Mrs. Hudson J C (Bessie Waldo Daniels)"
]]

# Notice that when there are multiple rows, the resulting data is stored in a `DataFrame`.
#
# The members of the Allison family happen to be consecutive rows of the `DataFrame`. If you want to select a consecutive set of rows, you do not need to type the index of every row that you want. Instead, you can use **slice notation**. The slice notation `a:b` allows you to select all rows from `a` to `b`. So another way we could have selected all four members of the Allison family is to write:

df_by_name.loc["Allison, Master. Hudson Trevor":"Allison, Mrs. Hudson J C (Bessie Waldo Daniels)"]

# This behavior of the slice may be surprising to you if you are a Python veteran. We will say more about this in a second.

# What if you wanted to inspect the 100th row of the `DataFrame`, but didn't know the index label for that row? You can use `.iloc` to **select by position** (in contrast to `.loc`, which **selects by label**).
#
# Remember that `pandas` (and Python in general) uses zero-based indexing, so the position index of the 100th row is 99.

df_by_name.iloc[99]

# You can also select multiple rows by position, either by passing in a list:

df_by_name.iloc[[99, 100]]

# or by using slice notation:

df_by_name.iloc[99:101]

# Notice the difference between how slice notation works for `.loc` and `.iloc`.
#
# - `.loc[a:b]` returns the rows from `a` up to `b`, _including_ `b`.
# - `.iloc[a:b]` returns the rows from `a` up to `b`, _not including_ `b`.
#
# So to select the rows in positions 99 and 100, we do `.iloc[99:101]` because we want the rows from position 99 up to 101, _not including 101_. This is consistent with the behavior of slices elsewhere in Python. For example, the slice `1:2` applied to a list returns one element, not two.

test = ["a", "b", "c", "d"]
test[1:2]

# ### What Makes a Good Index?
#
# Something odd happens if we look for "Mr. James Kelly" in this `DataFrame`. Although we only ask for one label, we get two rows back.

df_by_name.loc["Kelly, Mr. James"]

# This happened because there were two passengers on the Titanic named "James Kelly". In general, a good row index should uniquely identify observations in the data set. Names are often, but not always, unique. The best row indexes are usually IDs that are guaranteed to be unique.
#
# Another common row index is time. If each row represents a measurement in time, then it makes sense to have the date or the timestamp be the index.

# ## Variables

# Recall that **variables** are the columns in a tabular data set. They are the measurements that we make on each observation.

# ### Selecting Variables
#
# Suppose we want to select the `age` column from the `DataFrame` above. There are three ways to do this.

# 1\.  Use `.loc`, specifying both the rows and columns. (_Note:_ The colon `:` is Python shorthand for "all".)

df.loc[:, "age"]

# 2\. Access the column as you would a key in a `dict`.

df["age"]

# 3\. Access the column as an attribute of the `DataFrame`.

df.age

# Method 3 (attribute access) is the most concise. However, it does not work if the variable name contains spaces or special characters, begins with a number, or matches an existing attribute of `DataFrame`. For example, if `df` had a column called `head`, `df.head` would not return the column because `df.head` already means something else, as we have seen.

# Notice that the data structure used to store a single column is again a `Series`, not a `DataFrame`. So single rows and columns are stored in `Series`.

# To select multiple columns, you would pass in a _list_ of variable names, instead of a single variable name. For example, to select both the `age` and `sex` variables, we could do one of the following:

# +
# METHOD 1
df.loc[:, ["age", "sex"]].head()

# METHOD 2
df[["age", "sex"]].head()
# -

# Note that there is no way to generalize attribute access (Method 3 above) to select multiple columns.

# ### The Different Types of Variables

# There is a fundamental difference between variables like `age` and `fare`, which can be measured on a numeric scale, and variables like `sex` and `home.dest`, which cannot. 
#
# Variables that can be measured on a numeric scale are called **quantitative variables**. Just because a variable happens to contain numbers does not necessarily make it "quantitative". For example, consider the variable `survived` in the Titanic data set. Each passenger either survived or didn't. This data set happens to use 1 for "survived" and 0 for "died", but these numbers do not reflect an underlying numeric scale.
#
# Variables that are not quantitative but take on a limited set of values are called **categorical variables**. For example, the variable `sex` takes on one of two possible values ("female" or "male"), so it is a categorical variable. So is the variable `home.dest`, which takes on a larger, but still limited, set of values. We call each possible value of a categorical variable a "category". Although categories are usually non-numeric (as in the case of `sex` and `home.dest`), they are sometimes numeric. For example, the variable `survived` in the Titanic data set is a categorical variable with two categories (1 if the passenger survived, 0 if they didn't), even though those values are numbers. With a categorical variable, one common analysis question is, "How many observations are there in each category?".
#
# Some variables do not fit neatly into either category. For example, the variable `name` in the Titanic data set is obviously not quantitative, but it is not categorical either because it does not take on a limited set of values. Generally speaking, every passenger will have a different name (the two James Kellys notwithstanding), so it does not make sense to analyze the frequencies of different names, as one might do with a categorical variable. We will group variables like `name`, that are neither quantitative nor categorical, into an "other" category.
#
# Every variable can be classified into one of these three **types**: quantitative, categorical, or other. The type of the variable often dictates the kind of analysis we do and the kind of visualizations we make, as we will see later in this chapter. 
#
# `pandas` tries to infer the type of each variable automatically. If every value in a column (except for missing values) can be cast to a number, then `pandas` will treat that variable as quantitative. Otherwise, the variable is treated as categorical. To see the type that Pandas inferred for a variable, simply select that variable using the methods above and look for its `dtype`. A `dtype` of `float64` or `int64` indicates that the variable is quantitative.  For example, the `age` variable above had a `dtype` of `float64`, so it is quantitative. On the other hand, if we look at the `sex` variable,

df.sex

# its `dtype` is `object`, so `pandas` will treat it as a categorical variable. Sometimes, this check can yield surprises. For example, if you only looked the first few rows of `df`, you might expect `ticket` to be a quantitative variable. But if we actually look at its `dtype`:

df.ticket

# it appears to be an `object`. That is because there are some values in this column that contain non-numeric characters. For example:

df.ticket[9]

# As long as there is one value in the column that cannot be cast to a numeric type, the entire column will be treated as categorical, and the individual values will be strings (notice the quotes around even a number like 24160, indicating that `pandas` is treating it as a string). 

df.ticket[0]

# If you wanted `pandas` to treat this variable as quantitative, you can use the `to_numeric()` function. However, you have to specify what to do for values like `'PC 17609'` that cannot be converted to a number. The `errors="coerce"` option tells `pandas` to treat these values as missing (`NaN`).

pd.to_numeric(df.ticket, errors="coerce")

# If we wanted to keep this change, we would assign this column back to the original `DataFrame`, as follows:
#
# `df.ticket = pd.to_numeric(df.ticket, errors="coerce")`.
#
# But since `ticket` does not appear to be a quantitative variable, this is not actually a change we want to make.

# There are also categorical variables that `pandas` infers as quantitative because the values happen to be numbers. As we discussed earlier, the `survived` variable is categorical, but the values happen to be coded as 1 or 0. To force `pandas` to treat this as a categorical variable, you can cast the values to strings. Notice how the `dtype` changes:

df.survived.astype(str)

# In this case, this is a change that we actually want to keep, so we assign the modified column back to the `DataFrame`.

df.survived = df.survived.astype(str)

# ## Summary
#
# - Tabular data is stored in a data structure called a `DataFrame`.
# - Rows represent observations; columns represent variables.
# - Single rows and columns are stored in a data structure called a `Series`.
# - The row index should be a set of labels that uniquely identify observations.
# - To select rows by label, we use `.loc[]`. To select rows by (0-based) position, we use `.iloc[]`.
# - To select columns, we can use `.loc` notation (specifying both the rows and columns we want, separated by a comma), key access, or attribute access.
# - Variables can be quantitative, categorical, or other.
# - Pandas will try to infer the type, and you can check the type that Pandas inferred by looking at the `dtype`.

# # Exercises

# **Exercise 1.** Consider the variable `pclass` in the Titanic data set, which is 1, 2, or 3, depending on whether the passenger was in 1st, 2nd, or 3rd class. 
#
# - What type of variable is this: quantitative, categorical, or other? (_Hint:_ One useful test is to ask yourself, "Does it make sense to add up values of this variable?" If the variable can be measured on a numeric scale, then it should make sense to add up values of that variable.)
# - Did `pandas` correctly infer the type of this variable? If not, convert this variable to the appropriate type.

## YOUR CODE HERE
## BEGIN SOLUTION
print(df.head())
print("data type before",df.pclass.dtype)
df.pclass = df.pclass.astype("category")
print("data type after",df.pclass.dtype)
## END SOLUTION

# ## YOUR TEXT HERE
# ### BEGIN SOLUTION
# Should be categorical but it is numeric (int64 type). We should change it to categorical.
# ### END SOLUTION

# Exercises 2-7 deal with the Tips data set (`/data301/data/tips.csv`). You can learn more about this data set on the first page of [this reference](http://www.ggobi.org/book/chap-data.pdf).

# **Exercise 2.** Read in the Tips data set into a `pandas` `DataFrame` called `tips`.
#
# - What is the unit of observation in this data set?
# - For each variable in the data set, identify it as quantitative, categorical, or other, based on your understanding of each variable. Did `pandas` correctly infer the type of each variable?

## YOUR CODE HERE
## BEGIN SOLUTION
tips = pd.read_csv("/data301/data/tips.csv")
print(tips.head())
print(tips.dtypes)
## END SOLUTION

# ## YOUR TEXT HERE
# ### BEGIN SOLUTION
# Unit of observation is a single bill or trip to the restaurant.
#
# total_bill: numeric, tip: numeric, size is an integer numeric, and the rest are categorical. Though arguments could be made that smoker for instance could be binary. We could also make them explicitly categorical.
#
# Yes. Pandas has roughly inferred the correct type.
# ### END SOLUTION

# **Exercise 3.** Make the day of the week the index of the `DataFrame`.
#
# - What do you think will happen when you call `tips.loc["Thur"]`? Try it. What happens?
# - Is this a good variable to use as the index? Explain why or why not.

## YOUR CODE HERE
## BEGIN SOLUTION
tips = tips.set_index("day")
print(tips.head())
print(tips.loc["Thur"])
## END SOLUTION

# ## YOUR TEXT HERE
# ### BEGIN SOLUTION
# The index is reasonable depending on how you plan to study the data. Often we want our index to be unique, so in that case `day` is not appropriate.
# ### END SOLUTION

# **Exercise 4.** Make sure the index of the `DataFrame` is the default (i.e., 0, 1, 2, ...). If you changed it away from the default in the previous exercise, you can use `.reset_index()` to reset it.
#
# - How do you think `tips.loc[50]` and `tips.iloc[50]` will compare? Now try it. Was your prediction correct?
# - How do you think `tips.loc[50:55]` and `tips.iloc[50:55]` will compare? Now try it. Was your prediction correct?

# YOUR CODE HERE
## BEGIN SOLUTION
tips = tips.reset_index()
print("loc[50]")
print(tips.loc[50])
print("iloc[50]")
print(tips.iloc[50])
print("loc[50:55]")
print(tips.loc[50:55])
print("iloc[50:55]")
print(tips.iloc[50:55])
## END SOLUTION

# ## YOUR TEXT HERE
# ### BEGIN SOLUTION
# As you can see, there are some small differences of note for this dataset, which include the number of rows returned is one less with `iloc[50:55]`. In general, they would be really different because one is using the `index` and one is using the numerical indices.
# ### END SOLUTION

# **Exercise 5.** How do you think `tips.loc[50]` and `tips.loc[[50]]` will compare? Now try it. Was your prediction correct?

## YOUR CODE HERE
### BEGIN SOLUTION
print(tips.loc[50])
print(tips.loc[[50]])
print(type(tips.loc[50]),type(tips.loc[[50]]))
### END SOLUTION

# ## YOUR TEXT
# ### BEGIN SOLUTION
# As you can see, the biggest difference is the data type returned, which really matters for subsequent calls. Sometimes you would like the Series object and other times you want the data frame.
# ### END SOLUTION

# **Exercise 6.** What data structure is used to represent a single column, such as `tips["total_bill"]`? How could you modify this code to obtain a `DataFrame` consisting of just one column, `total_bill`?

## YOUR CODE HERE
### BEGIN SOLUTION
print(tips["total_bill"])
print(type(tips["total_bill"]))
print(tips[["total_bill"]])
print(type(tips[["total_bill"]]))
### END SOLUTION

# **Exercise 7.** Create a new `DataFrame` from the Tips data that consists of just information about the table (i.e., whether or not there was a smoker, the day and time they visited the restaurant, and the size of the party), without information about the check or who paid.
#
# (There are many ways to do this. How many ways can you find?)

## YOUR CODE HERE
### BEGIN SOLUTION
print("No single solution. Open ended.")
### END SOLUTION

# ### Reflection
# Based on your experiments with the labs this week, share something new that had not been mentioned in class such as a tidbit of information.
