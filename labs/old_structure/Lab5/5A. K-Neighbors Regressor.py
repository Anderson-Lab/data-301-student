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

# # K-Neighbors Regressor
#
# Your goal is to train a model to predict the bitterness of a beer (in International Bittering Units, or IBU), given features about the beer. You can acquire the data in any one of three places:
#
# - on [Kaggle](https://www.kaggle.com/c/beer2019/data) 
# - on [Github](https://github.com/dlsun/data-science-book/tree/master/data/beer)
# - in the `/data301/data/beer/` directory
#
# A description of the variables is available [here](https://www.kaggle.com/c/beer2019/data).

# # Question 1
#
# Choose at least 5 different _sets_ of features that you think might be important. For example, three different sets of features might be:
#
# - `abv`
# - `abv`, `available`, `originalGravity`
# - `originalGravity`, `srm`
#
# (You do not have to use these sets of features. They are provided just as an example.)
#
# For each set of features, train a $30$-nearest neighbor model to predict IBU (`ibu`). Determine which of these models is best at predicting IBU. Is it the model that contained the most features?

# +
# TYPE YOUR CODE HERE.
# -

# **SUMMARIZE YOUR OBSERVATIONS HERE.**

# # Question 2
#
# Let's see how the distance metric and the scaling method influence prediction accuracy. Use the set of features from Question 1 that you determined to be the best. Continue to use $k=30$ nearest neighbors, but try fitting models with different distance metrics and scaling methods. Which distance metric and/or scaling method gives the best prediction accuracy?

# +
# TYPE YOUR CODE HERE.
# -

# **SUMMARIZE YOUR OBSERVATIONS HERE.**

# # Question 3
#
# Now, we will determine the right value of $k$. Use the set of features, the distance metric, and the scaling method that you determined to be best in Questions 1 and 2. Fit $k$-nearest neighbor models for different values of $k$. Plot the training error and the test error as a function of $k$, and determine the optimal value of $k$.

# +
# TYPE YOUR CODE HERE.
# -

# **SUMMARIZE YOUR OBSERVATIONS HERE.**

# # Submission Instructions
#
# Once you are finished, follow these steps:
#
# 1. Restart the kernel and re-run this notebook from beginning to end by going to `Kernel > Restart Kernel and Run All Cells`.
# 2. If this process stops halfway through, that means there was an error. Correct the error and repeat Step 1 until the notebook runs from beginning to end.
# 3. Double check that there is a number next to each code cell and that these numbers are in order.
#
# Then, submit your lab as follows:
#
# 1. Go to `File > Export Notebook As > PDF`.
# 2. Double check that the entire notebook, from beginning to end, is in this PDF file. (If the notebook is cut off, try first exporting the notebook to HTML and printing to PDF.)
# 3. Upload the PDF [to PolyLearn](https://polylearn.calpoly.edu/AY_2018-2019/mod/assign/view.php?id=325687).
