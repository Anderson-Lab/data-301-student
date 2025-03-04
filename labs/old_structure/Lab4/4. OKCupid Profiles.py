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

# # OKCupid Profiles
#
# **This "lab" was last year's Exam 1.**
#
# In this lab, you will answer questions using public data about OKCupid users living in the San Francisco Bay Area. This data set can be found on JupyterHub at `/data301/data/okcupid/profiles.csv`, and the data documentation can be found at `/data301/data/okcupid/codebook.txt`.
#
# Some of the questions are deliberately ambiguous. I want to see how you take an ambiguous, real-world question and translate it into a concrete question that can be answered with data. Many answers are acceptable, as long as you do a good job justifying your answer. Just answer each question to the best of your ability according to _your_ interpretation of the question.

# +
# %matplotlib inline
import pandas as pd
import numpy as np

pd.options.display.max_rows = 30
# -

# ## Question 1 (3 points)
#
# What proportion of OKCupid users report never smoking? Explain how you came up with your answer.

# +
# TYPE YOUR CODE HERE.
# -

# **EXPLAIN YOUR ANSWER HERE.**

# ## Question 2 (3 points)
#
# Make a single visualization that displays and facilitates comparison of:
# - the distribution of ages of users who are currently in college/university
# - the distribution of ages of users who are currently in med school
#
# Interpret what you see.

# +
# TYPE YOUR CODE HERE.
# -

# **INTERPRET YOUR PLOT HERE.**

# ## Question 3 (3 points)
#
# There are 10 essay prompts. Each user can choose which prompts to respond to. Make a visualization that shows the proportion of users that responded to each essay prompt. Interpret what you see. (Refer to the codebook for information about the essay prompts.)

# +
# TYPE YOUR CODE HERE.
# -

# **INTERPRET YOUR PLOT HERE.**

# ## Question 4 (3 points)
#
# Make a visualization showing the conditional distributions of sexual orientation given sex. Interpret what you see.

# +
# TYPE YOUR CODE HERE.
# -

# **INTERPRET YOUR PLOT HERE.**

# ## Question 5 (3 points)
#
# Make a visualization that shows the average height, as a function of age and sex. Interpret what you see.
#
# _Hint:_ There are two outliers in the data set that you may want to remove to make this plot look better.

# +
# TYPE YOUR CODE HERE.
# -

# **INTERPRET YOUR PLOT HERE.**

# ## Question 6 (3 points)
#
# Make a bar chart showing the number of users with each type of job. Sort the jobs by average reported income. (No explanation necessary.)

# +
# TYPE YOUR CODE HERE.
# -

# ## Question 7 (3 points)
#
# Make a visualization that shows the distribution of the _number_ of languages spoken by OKCupid users. (No explanation necessary.)

# +
# TYPE YOUR CODE HERE.
# -

# ## Question 8 (3 points)
#
# In this group of users, which religion's adherents are the most serious about their religion? Explain your answer.

# +
# TYPE YOUR CODE HERE.
# -

# **EXPLAIN YOUR ANSWER HERE.**

# ## Question 9 (6 points)
#
# Study the profile of user 28555 in the data set, especially what they wrote in `essay9` about what they are looking for in a partner. Recommend five users in this data set that you think would be most compatible with this user.

# +
# TYPE YOUR CODE HERE.
# -

# **EXPLAIN YOUR METHODOLOGY AND YOUR ANSWER HERE.**

# # Submission Instructions
#
# You do not need to submit this lab. However, I encourage you to check that your code runs correctly from start to finish.
#
# 1. Restart the kernel and re-run this notebook from beginning to end by going to `Kernel > Restart Kernel and Run All Cells`.
# 2. If this process stops halfway through, that means there was an error. Correct the error and repeat Step 1 until the notebook runs from beginning to end.
# 3. Double check that there is a number next to each code cell and that these numbers are in order.
