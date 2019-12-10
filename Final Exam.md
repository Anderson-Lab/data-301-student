---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Final Exam: Predict X Technical Task

**Note:** This is a modification of a real application assignment given to aspiring junior data scientist for a company called PredictX. 

**WARNING** I will run you notebook from start to finish so make sure you don't have errors or extra code. You will want to restart your kernel and test everything before submitting.

**Points** All prompts are worth 5 points unless otherwise stated.

In this task we are trying to predict whether we can identfy any item as food or not food as provided in the is_food column. The task has been split up into the following parts

* Preliminary data exploration (get a feel for the data)
* Data cleaning (Checking for Nan)
* Detailed data exploration (identifying useful features)
* Machine learning 
    * Dividing data into test and train sets
    * Scaling variables
    * Evaluating model (ROC, F1, Precision, Recall)
* Clustering
* Conclusions and suggestions
    
We start by loading some modules that are useful:  

```python jupyter={"outputs_hidden": false}
# Data processing
import pandas as pd
import numpy as np

# Plotting
%matplotlib inline
```

```python
df = pd.read_csv('expenses.csv')
df.head(6)
```

## Preliminary data exploration


### Prompt: Fix the data such that the first column is the index and give that index the name 'index'

```python
# TYPE SOLUTION HERE
```

### Prompt: Summarize the number of entries in df

```python
# TYPE SOLUTION HERE
```

### Prompt: What are the number of unique values for each column? Make sure to not report the number of unique values for floating point columns.

```python
# TYPE SOLUTION HERE
```

### Prompt: What columns have nan values?

```python
# TYPE SOLUTION HERE
```

### Prompt: Count the number of rows that have nan values?

```python
# TYPE SOLUTION HERE
```

### Prompt: Number of nan values per column?

```python jupyter={"outputs_hidden": false}
# TYPE SOLUTION HERE
```

<!-- #region {"jupyter": {"outputs_hidden": false}} -->
### Prompt: Please summarize in words (bullet list prefered) what you can now say about the data
<!-- #endregion -->

***YOUR SOLUTION HERE***


## Data cleanup

### Prompt: Perform simple data cleanup:

* Removing any rows where is_food is null
* Filling in any other nulls with a string

```python
# TYPE YOUR SOLUTION HERE
```

## Detailed data exploration

The point of these prompts is to compare food and non-food


### Prompt: Create a histogram showing the counts for non-food and food in the data. Then print out the percentages

```python jupyter={"outputs_hidden": false}
# TYPE YOUR CODE HERE
```

### Prompt: Visually compare the distribution of 'amount' and 'amount_per_day' grouped by 'is_food'. Make sure the graph is zoomed in enough to make it useful. Make sure you put food and non-food on the same graph. HINT 1: There is a density function you can use. HINT 2: You can also zoom in using the ``xlim`` parameter.

```python jupyter={"outputs_hidden": false}
# TYPE YOUR CODE HERE
```

```python
# TYPE YOUR CODE HERE
```

### Prompt: Do you think there is anything odd about these distributions? In other words, might there be any errors in the data?

<!-- #region {"jupyter": {"outputs_hidden": false}} -->
***TYPE YOUR RESPONSE HERE***
<!-- #endregion -->

### Prompt: Visualize the distribution (hist) of ``country`` for both non-food and food. Limit your visualization to only the top 20 countries sorted by frequency. Do any countries stand out?

```python jupyter={"outputs_hidden": false}
# TYPE YOUR CODE HERE
```

### Prompt: Visualize the distribution (hist) of ``type`` for both non-food and food. Limit your visualization to only the top 20 types sorted by frequency. Do any types stand out?

```python jupyter={"outputs_hidden": false}
# TYPE YOUR CODE HERE
```

### Prompt (Hard(er) Question) (10 points): Visualize the distribution (hist or value_counts+bar) of ``description`` for both non-food and food using CountVectorizer. Limit your visualization to only the top 10 words sorted by frequency. Do any words stand out?

```python
# Useful code for how to get from a sparse scipy matrix to a dataframe.
# X_df = pd.DataFrame.sparse.from_spmatrix(X)
# X_df.columns = vectorizer.get_feature_names()
# np.nansum(x)
# Also, you'll might have memory issues if you don't do this one column at a time
# At one point I used barh(alpha=0.5,figsize=(10,6)) because I wanted a horizontal bar chart and a bigger figure so I could read the axis
```

```python
# TYPE YOUR CODE HERE
```

# End Exploration


## Clustering


### Prompt (10 points): Cluster the expenses into three clusters based on the top 20 words (frequency sorted) in the description column. Plese use ``binary == True`` in the constructor of CountVectorizer. Drop any expenses that don't have any of the top 20 words. Produce a visual that allows me (or anyone else) to visually compare the clusters and the words contained in those clusters. I'm happy to share the beginning of my code, but it will result in a maximum number of points awarded of 7/10. Come up and get a look if you want this "feature".

```python
# TYPE YOUR CODE HERE
```

# Predicting is_food
The point of this section is to create a sequence of better performing models that try to predict is_food. 

Code that could be useful at some point:

``from scipy.sparse import hstack``

``X = hstack((X1,X2))``

Another hint: Be careful using standard scaler on your entire dataset. Instead consider whether you need to scale or whether scaling the numeric columns only is a decent option.


### Prompt: Use DictVectorizer to vectorize all of the columns except is_food and description, store the results into a sparse scipy matirx called X.

```python
# TYPE YOUR CODE HERE
```

### Prompt: Convert X to a sparse dataframe called X_df using from_spmatrix(). Set the column names correctly.

```python
# TYPE YOUR CODE HERE
```

### Prompt: Split X_df and df['is_food'] into X_train, y_train, X_test, y_test where X_test is 33% of the data.

```python
# TYPE YOUR CODE HERE
```

### Prompt: Create a KNN classifier with n_neighbors=11 and report the precision and recall on the test dataset. This takes a moment to run, so you might want to sample the data during testing, and then let it run when you know it is working. Just a suggestion.

```python
# TYPE YOUR CODE HERE
```

### Prompt: Scale the ``amount`` and ``amount_per_day`` columns and store the results in X_train_sc and X_test_sc, and repeat the previous prompt. Did this help?

```python
# TYPE YOUR CODE HERE
```

### Prompt: Create two logistic regression classifiers. Train one with X_train and one with X_train_sc. Is there a performance difference? How does this compare with KNN above?

```python
# TYPE YOUR CODE HERE
```

### Prompt (10 points): Use GridSearchCV to search through the following parameters on LogisticRegression. Find the best parameters from this search and then create a classifier and test this last classifier on X_test. NOTE: To deal with an odd error, I had to call ``X_train.sparse.to_dense()`` and pass that to fit instead of just X_train. Did class_weight equal to 'balanced' improve the classification? If not, go ahead and run it manually and describe what you see happening.

```python
# TYPE YOUR CODE HERE
from sklearn.model_selection import GridSearchCV

# Create regularization penalty space
penalty = ['l1', 'l2']

class_weight = [{},'balanced']
```

### Prompt (open ended) (10 points): Achieve near perfect results by incorporating description. I had memory issues unless I limited the maximum number of features to 1000 for the count vectorizer (``max_features=1000``). 

```python
# TYPE YOUR CODE HERE
```

# JSON QUESTION (not related to above) (10 points)
Using json_normalize and no ``for`` loops, create a single bar chart with info from the 18th, 19th, and 20th century. Specifically, I want you to show me the percentage for each event type in the top 20 event types of the 20th century.

```python
# TYPE YOUR CODE HERE
from pandas.io.json import json_normalize
import json
with open("/data301/data/nyphil/complete.json") as f:
    nyphil = json.load(f)
    
```

# Happy Holidays! You are done with DATA 301!!! Well. Except the project...

```python

```
