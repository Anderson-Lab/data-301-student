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

# # Chapter 10 Textual Data
#
# You may not be used to thinking about _text_, like an e-mail or a newspaper article, as data. But just as we might want to predict the price of a home or group wines into similar types, we might want to predict the sender of an e-mail or group articles into similar types. To leverage the machine learning techniques we have already learned, we will need a way to convert raw text into tabular form. This chapter introduces some principles for doing this.

# # 10.1 Bag of Words and N-Grams
#
# In data science, a text is typically called a **document**, even though a document can be anything from a text message to a full-length novel.  A collection of documents is called a **corpus**. In this chapter, we will work with a corpus of text messages, which contains both spam and non-spam ("ham") messages.

# +
# %matplotlib inline
import pandas as pd
pd.options.display.max_rows = 10

texts = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/SMSSpamCollection.txt", 
    sep="\t",
    names=["label", "text"]
)
texts
# -

# We might, for example, want to train a classifier to predict whether or not a text message is spam. To use machine learning techniques like $k$-nearest neighbors, we have to transform each of these "documents" into a more regular representation.
#
# A **bag of words** representation reduces a document to just the multiset of its words, ignoring grammar and word order. (A _multiset_ is like a set, except elements are allowed to appear more than once.)
#
# So, for example, the **bag of words** representation of the string "I am Sam. Sam I am." would be `{I, I, am, am, Sam, Sam}`. In Python, it is easiest to represent multisets using dictionaries, where the keys are the (unique) words and the values are the counts. So we would represent the above bag of words as `{"I": 2, "am": 2, "Sam": 2}`.
#
# Let's convert the text messages to a bag of words representation. To do this, we will use the `Counter` object in the `collections` module of the Python standard library. First, let's see how the `Counter` works.

from collections import Counter
Counter(["I", "am", "Sam", "Sam", "I", "am"])

# It takes in a list and returns a dictionary of counts---in other words, the bag of words representation that we want. But to be able to use `Counter`, we have to first convert our text into a list of words. We can do this using the string methods in Pandas, such as `.str.split()`, which splits a string into a list based on some character (which, by default, is whitespace).

texts["text"].str.split()

# There are several problems with this approach:
#
# - **It is case-sensitive.**  The word "the" in message 5567 and the word "The" in message 5570 are technically different strings and will be treated as different words by the `Counter`.
# - **There is punctuation.**  For example, in message 0, one of the words is "point,". This will be treated differently from the word "point".
#
# We can **normalize** the text for case by 
#
# - converting all of the characters to lowercase, using the `.str.lower()` method
# - stripping punctuation using a regular expression. The regular expression `[^\w\s]` tells Python to look for any pattern that is not (`^`) either an alphanumeric character (`\w`) or whitespace (`\s`). That is, it will detect any occurrence of punctuation. We will then use the `.str.replace()` method to replace all detected occurrences with the empty string, effectively removing all punctuation from the string.
#
# By chaining these commands together, we obtain a list, to which we can apply the `Counter` to obtain the bag of words representation.

# +
words = (
    texts["text"].
    str.lower().
    str.replace("[^\w\s]", "").
    str.split()
)

words
# -

words.apply(Counter)

# ## CountVectorizer and Classification
# Now we can put several of these concepts together from this book to see how well we can predict spam. First, we want to understand our data and the underlying distribution of values.

(texts['label'].value_counts()/texts.shape[0]).plot.bar()

# What the above graph shows us is that the percent of spam in our corpus is less than 20% of the data. We need to keep this in mind during evaluation. 
#
# The first thing that we need to incorporate into our pipeline is a vectorizer that performs the counting we described above. In ``sklearn``, we use ``CountVectorizer``. 

texts['label'].value_counts()

# +
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

print(__doc__)

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', KNeighborsClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.5, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'clf__n_neighbors': [10,30]
}

#from sklearn.model_selection import StratifiedKFold
#skf = StratifiedKFold(n_splits=3)

grid_search = GridSearchCV(pipeline, parameters, cv=3,
                           n_jobs=-1, verbose=1,scoring='f1_macro')

grid_search.fit(texts['text'], texts['label'])
# -

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# **Notes about macro verus micro averaging** A macro-average will compute the metric independently for each class and then take the average (hence treating all classes equally), whereas a micro-average will aggregate the contributions of all classes to compute the average metric. 

# ## N-Grams
#
# The problem with the bag of words representation is that the ordering of the words is lost. For example, the following sentences have the exact same bag of words representation, but convey different meanings:
#
# 1. The dog bit her owner.
# 2. Her dog bit the owner.
#
# The first sentence has only two actors (the dog and its owner), but the second sentence has three (a woman, her dog, and the owner of something). To better capture the _semantic_ meaning of these two documents, we can use **bigrams** instead of individual words. A **bigram** is simply a pair of consecutive words. The "bag of bigrams" of the two sentences above are quite different:
#
# 1. {"The dog", "dog bit", "bit her", "her owner"}
# 2. {"Her dog", "dog bit", "bit the", "the owner"}
#
# They only share 1 bigram (out of 4) in common, even though they share the same 5 words.
#
# Let's get the bag of bigrams representation for the words above. To generate the bigrams from the list of words, we will use the `zip` function in Python, which takes in two lists and returns a single list of pairs (consisting of one element from each list):

list(zip([1, 2, 3], [4, 5, 6]))


# +
def get_bigrams(words):
    # We need to line up the words as follows:
    #   words[0], words[1]
    #   words[1], words[2]
    #       ... ,  ...
    # words[n-1], words[n]
    return zip(words[:-1], words[1:])

words.apply(get_bigrams).apply(Counter)
# -

# Instead of taking 2 words at a time, we could take 3, 4, or, in general, $n$ words. 
# A tuple of $n$ consecutive words is called an $n$-gram, and we can convert any document to a "bag of $n$-grams" representation. 
#
# The larger $n$ is, the better the representation will capture the meaning of a document. But if $n$ is so large that hardly any $n$-gram occurs more than once, then we will not learn much from this representation.

# # Exercises

# **Exercise 1.** Read in the OKCupid data set (`/data301/data/okcupid/profiles.csv`). You can get more information about this dataset at ``https://github.com/rudeboybert/JSE_OkCupid``. Convert the users' responses to `essay0` ("self summary") into a bag of words representation.
#
# (_Hint:_ Test your code on the first 100 users before testing it on the entire data set.)

# TYPE YOUR CODE HERE.
# BEGIN SOLUTION
import pandas as pd
okcupid = pd.read_csv('/data301/data/okcupid/profiles.csv')
okcupid.head()
# END SOLUTION

# **Exercise 2.** The text of _Green Eggs and Ham_ by Dr. Seuss can be found in (`https://raw.githubusercontent.com/dlsun/data-science-book/master/data/drseuss/greeneggsandham.txt`). Read in this file and convert this "document" into a bag of trigrams (3-grams) representation. Which trigram appears most often? Some code has been provided to get you started.

# TYPE YOUR CODE HERE.
with open(
    "/data301/data/drseuss/greeneggsandham.txt",
    "r"
) as f:
    for line in f:
        pass

# **Exercise 3 (Challenge Problem)** Consider the okcpid dataset. Train and hyperparameter tune a KNN model to predict the pet preferences.

okcupid.pets.value_counts()


