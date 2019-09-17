# -*- coding: utf-8 -*-
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

# # 1.2 Summarizing Variables
#
# In the previous section, we emphasized the difference between quantitative and categorical variables. The distinction is not merely pedantic; `pandas` will actually behave differently depending on whether it thinks a variable is quantitative or categorical.
#
# It is not easy for a human to make sense of _all_ the values of a variable. In this section, we focus on ways to reduce the values to just a handful of summary statistics. Our working example will again be the Titanic data set, which contains both quantitative and categorical variables.

# +
import pandas as pd
pd.options.display.max_rows = 8

df = pd.read_csv("/data301/data/titanic.csv")
df
# -

# To get a quick summary of a variable, we can use the `.describe()` function. Let's see what happens when we call `.describe()` on a quantitative variable, like `age`.

df.age.describe()

# It returns the count (the number of observations with non-missing values), the mean, the standard deviation (`std`), and various percentiles (`min`, `25%`, `50%`, `75%`, `max`).
#
# Now, what if we call `.describe()` on a categorical variable, like `embarked`? This is a variable that takes on the values `C`, `Q`, or `S`, depending on whether the passenger embarked at Cherbourg, Queenstown, or Southampton.

df.embarked.describe()

# The description of this variable is very different. We still get the count (of non-missing values). But instead of the mean and standard deviation (how would you calculate the mean of `Q` and `S`, anyway?), we get the number of unique values (`unique`), the value that appeared most often (`top`), and how often it appeared (`freq`). These are more natural summaries for a categorical variable, which only take on a limited set of values, where the values are often not even numeric.

# The `.describe()` function only provides a handful of the many summary statistics that are available in `pandas`. We extract additional summary statistics below.

# ## Summary Statistics for Quantitative Variables
#
# What statistics should we use to summarize a quantitative variable? The most salient features of a quantitative variable are its **center** and **spread**.
#
# ### Measures of Center
#
# Some statistics measure the **center** of a variable. Two commonly used measures of the center are:
#
# - the **mean** (a.k.a. average): the sum of the values divided by the count
# - the **median**: the middle value when you sort the values (i.e., a value such that 50% of the values lie below and 50% of the values lie above)
#
# A measure of center gives us information about the "typical" value of a variable. For example, you might not know whether a typical fare on the Titanic was £1, £10, or £100. But if we calculate the mean:

df.fare.mean()

# we see that a typical fare is around £30.
#
# Let's see what the median says about the "typical" fare:

df.fare.median()

# The median is quite different from the mean! It says that about 50% of the passengers paid less than £15 and about 50% paid more, so another reasonable value for the "typical" fare is £15.
#
# The mean was twice the median! What explains this discrepancy? The reason is that the mean is very sensitive to extreme values. To see this, let's look at the highest fare that any passenger paid.

df.fare.max()

# The highest fare paid was over £500! Even if most passengers paid less than £15, extreme values like this one will drag the mean upward. On the other hand, since the median is always the middle value, it is not affected by the extreme values, as long as the ordering of the values is not changed.
#
# To drive this point home, let's see what would happen to the mean and median if that maximum fare were actually £10,000.

fare_10k = df.fare.replace(df.fare.max(), 10000)
fare_10k.mean(), fare_10k.median()

# Notice how the mean is now over £60, but the median is unchanged.

# Just to satisfy our curiosity, let's learn more about this passenger who paid the maximum fare. To do this, we have to find the row that achieved this maximum value. Fortunately, there is a convenient `pandas` function, `.idxmax()`, that returns the _row index_ of the maximum fare. (A mathematician might call this the ["arg max"](https://en.wikipedia.org/wiki/Arg_max).)

df.fare.idxmax()

# Now we can select the row corresponding to this index using `.loc`, as we learned in the previous section.

df.loc[df.fare.idxmax()]

# The median is a number below which 50% of the values fall. What if we want to know some other percentile? We can use the `.quantile()` function, which takes a percentile rank (between 0 and 1) as input and returns the corresponding percentile.
#
# For example, the 75th percentile is:

df.fare.quantile(.75)

# which is pretty close to the mean. So only about 25% of the passengers paid more than the mean! The mean is not a great measure of center when there are extreme values, as in this data set.

# To summarize, we have encountered several `pandas` functions that can be used to summarize a quantitative variable:
#
# - `.mean()` calculates the mean or average.
# - `.median()` calculates the median.
# - `.quantile(q)` returns a value such that a fraction `q` of the values fall below that value (in other words, the (100q)th percentile).
# - `.max()` calculates the maximum value.
# - `.idxmax()` returns the index of the row with the maximum value. If there are multiple rows that achieve this value, then it will only return the index of the first occurrence.
#
# The corresponding functions for the _minimum_ value exist as well:
#
# - `.min()` calculates the minimum value.
# - `.idxmin()` returns the index of the row with the minimum value. If there are multiple rows that achieve this value, then it will only return the index of the first occurrence.

df.fare.min()

# Some passengers boarded the Titanic for free, apparently.

# ### Measures of Spread
#
# The center of a quantitative variable only tells part of the story. For one, it tells us nothing about how spread out the values are. Therefore, it is important to also report a measure of **spread**.
#
# Let's investigate a few measures of spread that are built into `pandas`. For completeness, the formulas for these statistics are provided, where $x_1, ..., x_n$ represent the values and $\bar x$ their mean. But don't worry to much about the formulas if you understand the intuition.
#
# The first statistic that might come to mind is the **mean absolute deviation**, or MAD. To calculate the MAD, you first calculate the difference between each observation and the mean. Values below the mean will have a negative difference, while values above the mean will have a positive difference. We don't want the negative differences to cancel out the positive differences, since _all_ of them contribute to the spread. So we take the absolute value of all the differences and then average.
#
# $$
# \begin{align*}
# \textrm{MAD} &= \textrm{mean of } |x_i - \bar x| \\
# &= \frac{1}{n} \sum_{i=1}^n |x_i - \bar x|
# \end{align*}
# $$
#
# We can implement the MAD ourselves using the `.mean()` and `.abs()` functions.

# STEP 1: Calculate the difference between each fare and the mean.
(df.fare - df.fare.mean())

# STEP 2: Calculate the absolute value of each difference.
(df.fare - df.fare.mean()).abs()

# STEP 3: Take the mean of these absolute differences.
(df.fare - df.fare.mean()).abs().mean()

# Notice that in Step 1, we subtracted a single value (`df.fare.mean()`) from a `pandas` `Series` (`df.fare`). A `Series` is like an array, and in most programming languages, subtracting a number from an array is a type mismatch. But `pandas`  automatically **broadcasted** the subtraction over each number in the `Series`.
#
# The `.abs()` function in Step 2 is another example of broadcasting. The absolute value function is applied to each element of the `Series`.

# The MAD is actually built into `pandas`, so there really is no reason to implement it from scratch, as we did above. Let's check that we get the same answer when we call the built-in function.

df.fare.mad()

# Since the MAD is a mean of the absolute differences and the mean represents the "typical" value, we can interpret the MAD as saying that the "typical" fare is about £30 away from the average.

# Another way to ensure that the negative and positive differences don't cancel is to square all the differences before averaging. This leads to the definition of **variance**.
#
# $$\textrm{Variance} = \textrm{mean of } (x_i - \bar x)^2$$
#
# We can implement the variance ourselves using the .mean() and power (`**`) functions. Again, notice how the subtraction and the power function are broadcast over the elements of the `Series`.

((df.fare - df.fare.mean()) ** 2).mean()

# Alternatively, we can simply call the `.var()` function in `pandas`.

df.fare.var()

# You might be surprised that `.var()` produces a slightly different number. This is because `pandas` divides by $n-1$ in calculating the mean of the squared differences, rather than $n$. That is, the formula that `pandas` uses is 
#
# $$\text{Variance} = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar x)^2.$$
#
# To force Pandas to divide by $n$, you can set `ddof=0`.

df.fare.var(ddof=0)

# Now the value returned by `pandas` matches the value we obtained manually.

# #### Why We Divide By $n-1$ (Optional)
#
# Data is often a sample from some population. The point of calculating the variance of a sample is to be able to say something about the spread of the population.
#
# To see why we divide by $n-1$ to measure the spread of a population, consider the extreme case where we have a sample of size $n=1$. What can we say about the spread of the population based on this one observation? Absolutely nothing! We need a sample of size at least $n=2$ to be able to say anything about the _spread_. Therefore, the variance is undefined when $n=1$. In order to make the variance not defined for $n=1$, we divide by $n-1$ so that we have $0/0$ when $n=1$. The variance formula above is only defined when $n \geq 2$.

# The trouble with variance is that its units are wrong. If the original values $x_1, ..., x_n$ were in pounds, the variance would be in pounds _squared_. This is obvious if you simply look at the magnitude of the variance in the example above; the variance is in the _thousands_, even though the largest fare is just over £500.  To correct the units of variance, we take the square root to obtain a more interpretable measure of spread, called the **standard deviation**:
#
# $$\textrm{SD} = \sqrt{\textrm{Variance}}.$$

df.fare.std()

# We can interpret this standard deviation as saying: the "typical" fare is about £50 away from the average.

# The standard deviation is the most widely used measure of spread, more common than the MAD. At first, this might seem odd. To calculate the standard deviation, we squared the differences from the mean, only to take a square root in the end. Why bother with this rigmarole, when we could just calculate absolute values instead?
#
# The reasons for preferring the standard deviation are complicated. But the short answer is that the variance (which is the square of the standard deviation) is much nicer mathematically. If you know calculus, you might remember that the absolute value function does not have a derivative at 0. Therefore, the MAD is not _differentiable_, which makes it inconvenient mathematically. That doesn't necessarily mean that it's any worse as a measure of spread.

# ## Summary Statistics for Categorical Variables
#
# Although there are many ways to summarize a quantitative variable, there is really only one way to summarize a categorical variable. Since a categorical variable can only take on a limited set of values, we can completely summarize the variable by reporting the frequencies of the different categories. The `pandas` function that produces this summary is `.value_counts()`.

embarked_counts = df.embarked.value_counts()
embarked_counts

# Note that the counts are sorted in decreasing order by default, so the first element corresponds to `top` in the summary produced by `.describe()`. Southampton was the most common point of embarkation. 
#
# Since the counts are stored in a `pandas` `Series` indexed by category, we can extract a particular count using either label-based or position-based selection:

embarked_counts.loc["C"], embarked_counts.iloc[1]

# Instead of the _number_ of passengers embarking at each location, we might instead want to know the _percentage_ of passengers. To do this, divide the `Series` by the sum to turn the counts into **proportions**. (The term _proportion_ refers to a percentage when it is expressed as a number between 0 and 1, instead of between 0% and 100%.) Proportions must add up to 1, just as percentages must add up to 100%.

embarked_counts / embarked_counts.sum()

# Notice the use of _broadcasting_ again; `embarked_counts` is a `Series`, but `embarked_counts.sum()` is a number. When a `Series` is divided by a number, the division is automatically applied to each element of the `Series`, producing another `Series`.

# ### Binary Categorical Variables
#
# A binary categorical variable (i.e., a categorical variable with two categories) can be represented as a quantitative variable by coding one category as 1 and the other as 0.
#
# In the Titanic data set, the `survived` variable has been coded this way. Each passenger either survived (1) or didn't (0).

df.survived

# Although we can use `.value_counts()` to determine how many passengers survived:

df.survived.value_counts()

# we can also call `.sum()` and `.mean()` on this variable because the values are numeric.
#
# What does `.sum()` do?

df.survived.sum()

# `.sum()` returns the _number_ of ones. To see why, remember that this `Series` only 0s and 1s. Each 1 we encounter increments the sum by one, and each 0 contributes nothing to the sum. So when we add up all the numbers, we end up with the number of ones---or, in this example, the number of survivors.
#
# What about `.mean()`?

df.survived.mean()

# `.mean()` returns the _proportion_ of ones. To see why, remember that the mean is the sum divided by the number of observations. The sum, as we have just discussed, is the number of 1s. Dividing this by the number of observations gives us the proportion of 1s---or, in this example, the proportion of survivors.
#
# $$ \text{mean} = \frac{\text{sum}}{n} = \frac{\text{number of survivors}}{\text{number of passengers}} = \text{proportion of passengers who survived}.$$

# ## Summary Statistics for Other Variables?
#
# In the last section, we noted that `name` is not a categorical variable because it does not take on a limited set of values. Hopefully, you now see why it was important to make this distinction. It does not make sense to analyze `name` like we analyzed `embarked` above. For example, if we calculate the frequency of each unique value in `name`, we don't learn much, since names generally do not repeat. 

df.name.value_counts()

# That is why `name` was classified as an "other" variable. "Other" variables require additional processing before they can be summarized and analyzed. For example, if we extracted just the surnames from the `name` variable, then it might make sense to analyze this new variable as a categorical variable. The following case study shows how.
#
# ### Case Study: Extracting the Surname from the Names
#
# We can extract the surnames from the names using the [built-in string processing functions](https://pandas.pydata.org/pandas-docs/stable/text.html), all of which are preceded by `.str`. The string processing function that will be most useful to us is `.str.split()`, which allows us to split each string in a `Series` by some sequence of characters.  (In other words, the `split()` function is _broadcast_ over the strings in the `Series`.) Since the surname and other names are separated by `", "`, we will split by `", "` to obtain two chunks, the first of which is the surname.

df.name.str.split(", ")

# We can specify the option `expand=True` to get a `DataFrame` where each chunk is a separate column. The surnames are now in the first column.

df.name.str.split(", ", expand=True)

# Now we can select the surnames column (the column is named `0` in the `DataFrame`).

surnames = df.name.str.split(", ", expand=True)[0]
surnames

# Since there are multiple passengers with the same surname, this is a categorical variable. We can use `.value_counts()` to find out which surnames were most common.

surnames.value_counts()

# Don't worry if the string processing is a bit over your head at this point. The purpose of this example was to illustrate how "other" variables can be wrangled into a form amenable to analysis.

# ## Summary
#
# - Quantitative and categorical variables are summarized differently.
# - For quantitative variables, we typically report a measure of center (e.g., mean, median, quantiles) and a measure of spread (e.g., variance, standard deviation, MAD).
# - For categorical variables, we typically report the frequencies of the various categories, either as counts or as proportions.
# - Other variables require additional processing before they can be analyzed.

# # Exercises
#
# All of the following exercises use the Tips data set (`/data301/data/tips.csv`).

# **Exercise 1.** How many people were in the largest party served by the waiter? The smallest?

# +
# YOUR CODE HERE
# -

# **Exercise 2.** How could you use the `.quantile()` function to calculate the median? Check that your method works on an appropriate variable from the Tips data set.

# +
# YOUR CODE HERE
# -

# **Exercise 3.** Another measure of spread is the **interquartile range**, or IQR, defined as:
#
# $$ \textrm{IQR} = \textrm{75th percentile} - \textrm{25th percentile}. $$
#
# Measure the spread in the total bills by reporting the IQR.

# +
# YOUR CODE HERE
# -

# **Exercise 4.** Some people use MAD to refer to the **median absolute deviation**. The median absolute deviation is the same as the mean absolute deviation, but it uses the median instead of the mean:
#
# $$\textrm{M(edian)AD} = \textrm{median of } |x_i - \textrm{median}|. $$
#
# Calculate the median absolute deviation of the total bills. (The median absolute deviation is not built into Pandas, so you will have to implement it from scratch.)

# +
# YOUR CODE HERE
# -

# **Exercise 5.** Who pays the bill more often: men or women?

# +
# YOUR CODE HERE
