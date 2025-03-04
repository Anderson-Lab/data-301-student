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

# # 7B. Movie Ratings
#
# The MovieLens data set (`/data301/data/ml-1m/`) is a collection of 1 million movie ratings submitted by users that you worked with in Chapter 11. The information about the movies, ratings, and users are stored in three separate files: `movies.dat`, `ratings.dat`, and `users.dat`. To save time, we will work with only a random sample of only 10000 ratings stored in `ratings_small.dat`.
#
# The code below reads in the three data sets into `DataFrame`s. Refer to the data documentation (`/data301/data/ml-1m/README`) for information about what the columns mean.

# +
import pandas as pd
pd.options.display.max_rows = 15

ratings = pd.read_csv("/data301/data/ml-1m/ratings_small.dat",
                      sep="::",
                      engine="python",
                      header=None,
                      names=["UserID", "MovieID", "Rating", "Timestamp"])
users = pd.read_csv("/data301/data/ml-1m/users.dat",
                    sep="::",
                    engine="python",
                    header=None,
                    names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
movies = pd.read_csv("/data301/data/ml-1m/movies.dat",
                     engine="python",
                     sep="::",
                     header=None,
                     names=["MovieID", "Title", "Genres"])

movies
# -

# You will train a model to predict the rating that a user gives a movie, given information about the movie (genres) and the user (gender, age, and occupation).
#
# The code below converts the "Genres" column of `movies` into a `DataFrame` of dummy variables, with one column per genre. Feel free to use this code, free of charge.

# +
from sklearn.preprocessing import MultiLabelBinarizer

binarizer = MultiLabelBinarizer()
genres = pd.DataFrame(
    binarizer.fit_transform(movies["Genres"].str.split("|")),
    columns=binarizer.classes_
)

genres
# -

# Now, train a $k$-nearest neighbors model to predict the rating using the user's gender, age, and occupation, as well as the movie's genres. Find the optimal value of $k$, and report an estimate of the test error of this model.
#
# _Hint:_ This will require merging the ratings data with the users data and the movie genres information from above. The instructor or TAs can help you merge the data, for a 3 point penalty.

# +
# ENTER YOUR CODE HERE.
# -

# **SUMMARIZE YOUR FINDINGS HERE.**

# # Submission Instructions
#
# You do not need to submit this lab. However, I encourage you to check that your code runs correctly from start to finish.
#
# 1. Restart the kernel and re-run this notebook from beginning to end by going to `Kernel > Restart Kernel and Run All Cells`.
# 2. If this process stops halfway through, that means there was an error. Correct the error and repeat Step 1 until the notebook runs from beginning to end.
# 3. Double check that there is a number next to each code cell and that these numbers are in order.
