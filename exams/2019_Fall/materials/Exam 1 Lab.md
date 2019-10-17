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

# Exam 1 Lab - DATA 301 Fall 2019
$$\noindent$$**Instructor**: Paul Anderson
$$\noindent$$**Your name**:

$$\noindent$$**Instructions**: Do NOT under any circumstances attempt to communicate with someone else in the class about the contents of this exam. Doing so will result in an automatic 0


**Dataset**: The dataset contains information about NBA players from 2014-2017. I have provided some starter code below.

```python
%matplotlib inline
```

```python
import pandas as pd
import numpy as np
```

```python
df = pd.read_csv('stats_14-17.csv',index_col=0)
positions = pd.read_csv('positions_14-17.csv',index_col=0)[["Player","labels"]]
```

```python
positions.head(2)
```

```python
df.head(2)
```

## Set the index to the player name for both ``df`` and ``positions``.

```python
# YOUR SOLUTION HERE
```

```python
# YOUR SOLUTION HERE
```

## Before joining these two datasets, find and remove any players that have more than 1 observation in the dataset.

```python
# YOUR SOLUTION HERE
```

### Here is the code to join them together

```python
df = df.join(positions).drop("url",axis=1)
```

```python
df.head(2)
```

## In this new dataset, how many players are there for each label in ``labels`` and for each position in ``Pos``?

```python
# YOUR SOLUTION HERE
```

## Create a bar chart that shows the fraction of each position in ``Pos`` that is ``Active``.

```python
# YOUR SOLUTION HERE
```

## What is the correlation between all numerical columns in the dataframe ``df``? Apart from the trivial answer of the diagonal (correlation of a variable with itset), what two variables have the highest correlation?

```python
# YOUR SOLUTION HERE
```

## What is that correlation? Don't just type it. Access it from your previous result.

```python
# YOUR SOLUTION HERE
```

## Create a pivot table that averages over all possible positions in ``Pos`` and values of ``labels`` for each numeric variable.

```python
# YOUR SOLUTION HERE
```

# Consider the following dataframe with only numeric values remaining.

```python
X = df.drop(['Pos', 'G', 'Player_ID', 'Status', 'labels'], axis=1)
```

## Scale the data using StandardScaler from sklearn

```python
# YOUR SOLUTION HERE
```

## Construct a PCA model using the PCA class in sklearn using only 2 components. Fit to your scaled data and transform the scaled data into a numpy array called ``scores``.

```python
# YOUR SOLUTION HERE
```

## HINT: Here is code that will take the scores array and set up a scores dataframe with well named columns.

```python
scores_df = pd.DataFrame(scores,columns=["PC1","PC2"])
scores_df_joined = scores_df.join(df.reset_index())
scores_df_joined.head()
```

## Visualize all of the players in our new dimensions (PC1 and PC2). Color the graph by ``Pos``. Size the points by ``%FGA_DUNK``. 

```python
from altair import * 

Chart(scores_df_joined).mark_point().encode(x="PC1",y="PC2",color="Pos",size="%FGA_DUNK")
```

### Using your own approach explore the PCA results. You may use graphs. You may dig into some subset of the data. Use only those topics taught in the lass. Nothing random from Google.

```python
# BEGIN YOUR OPEN ENDED RESPONSE HERE
```

```python

```
