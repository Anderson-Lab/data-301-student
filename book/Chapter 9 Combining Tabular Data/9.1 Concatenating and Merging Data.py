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

# # Chapter 11. Combining Tabular Data
#
# In many situtions, the information you need is spread across multiple data sets, so you will need to combine multiple data sets into one. In this chapter, we explore how to combine information from multiple (tabular) data sets.
#
# As a working example, we will use the baby names data collected by the Social Security Administration. Each data set in this collection contains the names of all babies born in the United States in a particular year. This data is [publicly available](https://www.ssa.gov/OACT/babynames/limits.html), and a copy has been made available at `/data301/data/names/`.

# !ls /data301/data/names

# # 11.1 Concatenating and Merging Data
#
# # Concatenation
#
# Sometimes, the _rows_ of data are spread across multiple files, and we want to combine the rows into a single data set. The process of combining rows from different data sets is known as **concatenation**. Visually, to concatenate two or more `DataFrame`s means to stack them on top of one another.
#
# ![](concatenate.png)
#
# For example, suppose we want to understand how the popularity of different names evolved between 1995 and 2015. The 1995 names and the 2015 names are stored in two different files: `yob1995.txt` and `yob2015.txt`, respectively. To carry out this analysis, we will need to combine these two data sets into one.

# +
import pandas as pd
pd.options.display.max_rows = 10

names1995 = pd.read_csv("http://github.com/dlsun/data-science-book/blob/master/data/names/yob1995.txt?raw=true",
                        header=None,
                        names=["Name", "Sex", "Count"])
names1995
# -

names2015 = pd.read_csv("http://github.com/dlsun/data-science-book/blob/master/data/names/yob2015.txt?raw=true",
                        header=None,
                        names=["Name", "Sex", "Count"])
names2015

# To concatenate the two, we use the `pd.concat()` function, which accepts a _list_ of `pandas` objects (`DataFrames` or `Series`) and concatenates them.

pd.concat([names1995, names2015])

# There are two problems with the combined data set above. First, there is no longer any way to distinguish the 1995 data from the 2015 data. To fix this, we can add a "Year" column to each `DataFrame` before we concatenate. Second, the indexes from the individual `DataFrame`s have been preserved. (To see this, observe that the last index in the `DataFrame` is 32,951, which corresponds to the number of rows in `names2015`, but there are actually 59,032 rows in the `DataFrame`.) That means that there are two rows with an index of 0, two rows with an index of 1, and so on. To force `pandas` to create a completely new index for this `DataFrame`, ignoring the indices from the individual `DataFrame`s, we specify `ignore_index=True`.

names1995["Year"] = 1995
names2015["Year"] = 2015
names = pd.concat([names1995, names2015], ignore_index=True)
names

# Now this is a `DataFrame` that we can use!
#
# Notice that the data is currently in tabular form, with one row per combination of name, sex, and year. It makes sense to set these to be the index of our `DataFrame`.

names.set_index(["Name", "Sex", "Year"], inplace=True)
names

# We may want to show the counts for the two years side by side. In other words, we want a data cube with (name, sex) along one axis and year along the other. To do this, we can `.unstack()` the year from the index, just as we did in Chapter 2.

names.unstack("Year")

# The `NaN`s simply indicate that there were no children (more precisely, if you read [the documentation](https://www.ssa.gov/OACT/babynames/limits.html), fewer than five children) born in the United States in that year. In this case, it makes sense to fill these `NaN` values with 0.

names.unstack().fillna(0)

# # Merging (a.k.a. Joining)
#
# More commonly, the data sets that we want to combine actually contain different information about the same observations. In other words, instead of stacking the `DataFrame`s on top of each other, as in concatenation, we want to stack them next to each other. The process of combining columns or variables from different data sets is known as **merging** or **joining**.
#
# ![](merge.png)
#
# The observations in the two data sets may not be in the same order, so merging is not as simple as stacking the `DataFrame`s side by side. For example, the process might look as follows:
#
# ![](one-to-one.png)
#
# In _pandas_, merging is accomplished using the `.merge()` function. We have to specify the variable(s) that we want to match across the two data sets. For example, to merge the 1995 names with the 2015 names, we have to join on name and sex.

names1995.merge(names2015, on=["Name", "Sex"])

# The variables `Name` and `Sex` that we joined on each appear once in the resulting `DataFrame`. The variable `Count`, which we did not join on, appears twice---since there are columns called `Count` in both `DataFrame`s. Notice that `pandas` automatically appended the suffix `_x` to the name of the variable from the left data set and `_y` to the name from the right. We can customize the suffixes by specifying the `suffixes=` argument.

names1995.merge(names2015, on=["Name", "Sex"], suffixes=("1995", "2015"))

# In the code above, we assumed that the columns that we joined on had the same names in the two data sets. What if they had different names? For example, suppose the columns had been lowercase in one and uppercase in the other. We can specify which variables to use from the left and right data sets using the `left_on=` and `right_on=` arguments.

# +
# Create new DataFrames where the column names are different
names1995_lower = names1995.copy()
names2015_upper = names2015.copy()
names1995_lower.columns = names1995.columns.str.lower()
names2015_upper.columns = names2015.columns.str.upper()

# This is how you merge them.
names1995_lower.merge(
    names2015_upper,
    left_on=("name", "sex"),
    right_on=("NAME", "SEX")
)
# -

# What if the "variables" that we want to join on are in the index? We can always call `.reset_index()` to make them columns, but we can also specify the arguments `left_index=True` or `right_index=True` to force `pandas` to use the index instead of columns.

names1995_idx = names1995.set_index(["Name", "Sex"])
names1995_idx

names1995_idx.merge(names2015, left_index=True, right_on=("Name", "Sex"))

# Note that this worked because the left `DataFrame` had an index with two levels, which were joined to two columns from the right `DataFrame`.

# # One-to-One and Many-to-One Relationships
#
# In the example above, there was at most one (name, sex) combination in the 2015 data set for each (name, sex) combination in the 1995 data set. These two data sets are thus said to have a **one-to-one relationship**. Another example of a one-to-one data set is the Beatles example from above. Each Beatle appears in each data set exactly once, so the name is uniquely identifying.
#
# ![](one-to-one.png)
#
# However, two data sets need not have a one-to-one relationship. For example, a data set that specifies the instrument(s) that each Beatle played would potentially feature each Beatle multiple times (if they played multiple instruments). If we joined this data set to the "Beatles career" data set, then each row in the "Beatles career" data set would be mapped to several rows in the "instruments" data set. These two data sets are said to have a **many-to-one relationship**.
#
# ![](many-to-one.png)

# # Many-to-Many Relationships: A Cautionary Tale
#
# In the baby names data, the name is not uniquely identifying. For example, there are both males and females with the name "Jessie".

# +
jessie1995 = names1995[names1995["Name"] == "Jessie"]
jessie2015 = names2015[names2015["Name"] == "Jessie"]

jessie1995
# -

# That is why we have to be sure to join on both name and sex. But what would go wrong if we joined these two `DataFrame`s on just "Name"? Let's try it out:

jessie1995.merge(jessie2015, on=["Name"])

# We see that Jessie ends up appearing four times.
#
# - Female Jessies from 1995 are matched with female Jessies from 2015. (Good!)
# - Male Jessies from 1995 are matched with male Jessies from 2015. (Good!)
# - Female Jessies from 1995 are matched with male Jessies from 2015. (Huh?)
# - Male Jessies from 1995 are matched with female Jessies from 2015. (Huh?)
#
# The problem is that there were multiple Jessies in the 1995 data and multiple Jessies in the 2015 data. We say that these two data sets have a **many-to-many relationship**.

# # Exercises

# **Exercise 1.** Make a line plot showing the popularity of your name over the years. How popular was your name in the year you were born? 
#
# (If you have a rare name that does not appear in the data set, choose a friend's name.)

# +
# TYPE YOUR CODE HERE
# -

# Exercises 2-4 deal with the Movielens data (`/data301/data/ml-1m/`), which is a collection of movie ratings submitted by users. The information about the movies, ratings, and users are stored in three separate files, called `movies.dat`, `ratings.dat`, and `users.dat`. The column names are not included with the data files. Refer to the data documentation (`/data301/data/ml-1m/README`) for the column names and how the columns correspond across the data sets.

# **Exercise 2.** Who's more generous with ratings: males or females? Calculate the average of the ratings given by male users, and the average of the ratings given by female users.

# +
# TYPE YOUR CODE HERE
# -

# **Exercise 3.** Among movies with at least 100 ratings, which movie had the highest average rating?

# +
# TYPE YOUR CODE HERE
# -

# **Exercise 4.** For each movie, calculate the average age of the users who rated it and the average rating. Make a scatterplot showing the relationship between age and rating, with each point representing a movie. (Optional: Use the size of each point to represent the number of users who rated the movie.)

# +
# TYPE YOUR CODE HERE
