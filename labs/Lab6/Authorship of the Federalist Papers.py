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

# # Authorship of the Federalist Papers
#
# The _Federalist Papers_ were a set of 85 essays published between 1787 and 1788 to promote the ratification of the United States Constitution. They were originally published under the pseudonym "Publius". Although the identity of the authors was a closely guarded secret at the time, most of the papers have since been conclusively attributed to one of Hamilton, Jay, or Madison. The known authorships can be found in `/data301/data/federalist/authorship.csv`.
#
# For 15 of the papers, however, the authorships remain disputed. (These papers can be identified from the `authorship.csv` file because the "Author" field is blank.) In this analysis, you will train a classifier on the papers with known authorships and use your classifier to predict the authorships of the disputed papers. The text of each paper can be found in the `/data301/data/federalist/` directory. The name of the file indicates the number of the paper.

# ## Question 1
#
# When analyzing an author's style, common words like "the" and "on" are actually more useful than rare words like "hostilities". That is because rare words typically signify context. Context is useful if you are trying to find documents about similar topics, but not so useful if you are trying to identify an author's style because different authors can write about the same topic. For example, both Dr. Seuss and Charles Dickens used rare words like "chimney" and "stockings" in _How the Grinch Stole Christmas_ and _A Christmas Carol_, respectively. But they used common words very differently: Dickens used the word "upon" over 100 times, while Dr. Seuss did not use "upon" at all.
#
# Read in the Federalist Papers. Convert each one into a vector of term frequencies. In order to restrict to common words, include only the top 50 words. Then, train a $k$-nearest neighbors model on the documents with known authorship. Determine an optimal value of $k$ (it's up to you to decide what's "optimal"). 
#
# Report an estimate of the test accuracy, precision, and recall of your model.

# +
# TYPE YOUR CODE HERE.
# -

# **SUMMARIZE WHAT YOU DISCOVERED HERE.**

# ## Question 2
#
# What if we used TF-IDF on the top 50 words instead of the term frequencies? Repeat Question 1, using TF-IDF instead of TF. Which approach is better: TF-IDF or TF?

# +
# TYPE YOUR CODE HERE.
# -

# **SUMMARIZE WHAT YOU DISCOVERED HERE.**

# ## Question 3
#
# Using the model that you determined to be best in Questions 1 and 2, fit a $k$-nearest neighbors model to all 70 documents with known authorship. Create a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) for your model that shows how often you predicted Hamilton, Jay, or Madison, and how often it actually was Hamilton, Jay, or Madison (on the training data, of course). 
#
# From your confusion matrix, you should be able to calculate the (training) precision and recall of your model for predicting Hamilton. What is it?

# +
# TYPE YOUR CODE HERE.
# -

# **SUMMARIZE WHAT YOU DISCOVERED HERE.**

# ## Question 4
#
# Finally, use the model you trained in Question 3 to predict the authorships of the 15 documents with unknown authors. Summarize what you find.

# +
# TYPE YOUR CODE HERE
# -

# **SUMMARIZE WHAT YOU DISCOVERED HERE.**

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
# 3. Upload the PDF [to PolyLearn](https://polylearn.calpoly.edu/AY_2018-2019/mod/assign/view.php?id=336786).
