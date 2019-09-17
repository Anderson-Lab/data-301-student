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

# # Retrieval Practice and Learning
#
# What is the most effective way to learn a subject? Many students focus exclusively on the _encoding_ process---that is, how to get the knowledge into memory in the first place. For example, taking notes is an activity for encoding knowledge.
#
# _Retrieval_, on the other hand, is the process of reconstructing that knowledge from memory. [Karpicke and Blunt](http://science.sciencemag.org/content/331/6018/772) (2011) demonstrated that _retrieval_ is more effective for learning than activites designed to promote effective encoding. They conducted an experiment in which subjects had to learn about sea otters by reading a passage. Subjects were randomly assigned to one of two conditions: some were instructed to create a [concept map](https://en.wikipedia.org/wiki/Concept_map) as they read the passage, while others were instructed to practice retrieval (i.e., read the passage, recall as much as they could, read the text again, and recall again). The two main measurements they recorded were:
#
# 1. each subject's score on a follow-up learning test one week later
# 2. each subject's _prediction_ of how well they would do on that test
#
# In this lab, you will analyze data from a _replication_ of Karpicke and Blunt's experiment, conducted by Buttrick _et al_.
# - The data file is here: https://raw.githubusercontent.com/dlsun/data-science-book/master/data/KarpickeBlunt2011Replication/data.csv.
# - The codebook (explaining what the variables mean) is here: https://raw.githubusercontent.com/dlsun/data-science-book/master/data/KarpickeBlunt2011Replication/codebook.csv.

# +
# READ IN THE DATA SET HERE
# -

# # Question 1
#
# Which group felt like they learned more: the subjects who made concept maps or the ones who practiced retrieval? (Or are they about the same?) Make an appropriate visualization and explain what you see.
#
# _Hint:_ Use the variable `PR.2`, which contains the participants' predictions of how well they would do on a test one week later. 

# +
# YOUR CODE HERE
# -

# **YOUR EXPLANATION HERE**

# # Question 2
#
# Which group actually did better on the follow-up learning test one week later? Make an appropriate visualization and explain what you see.
#
# _Hint:_ Don't ask which variable you should use. That is for you to figure out. Read the codebook carefully (consulting the [original paper](http://science.sciencemag.org/content/331/6018/772), if necessary), make an informed decision, and explain your choice.

# +
# YOUR CODE HERE
# -

# **YOUR EXPLANATION HERE**

# # Question 3
#
# How good were subjects at predicting how well they would do on the follow-up learning test? Calculate a measure of how well subjects predicted their performance and interpret the value in context. (Optionally, you may want to include a visualization as well.)

# +
# YOUR CODE HERE
# -

# **YOUR EXPLANATION HERE**

# # Question 4
#
# This was a completely randomized experiment. This means that the condition that each subject was assigned to should be independent of their gender, age, and any other subject characteristics. Does that seem to be true in this case? Calculate a summary measure and/or make a visualization, and explain what you see.

# +
# YOUR CODE HERE
# -

# **YOUR EXPLANATION HERE**

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
# 3. Upload the PDF [to PolyLearn](https://polylearn.calpoly.edu/AY_2018-2019/mod/assign/view.php?id=313950).
