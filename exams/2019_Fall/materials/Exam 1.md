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

# Exam 1 - DATA 301 Fall 2019
$$\noindent$$**Instructor**: Paul Anderson
$$\noindent$$**Your name**:

$$\noindent$$**Instructions**: Please answer all questions as partial credit is always an option. Short and correct answers are preferred. Unless otherwise stated, you MUST show all of your work. It goes without saying, but just so it's official. No ``for`` loops allowed. And you must use what we have discussed in this course.
$$\noindent$$Please ignore ``%%capture``. It is only necessary for rendering this in PDF format.
$$\noindent$$Unless otherwise stated each question is worth 5 points.
$$\noindent$$Unless otherwise stated the dataframe ``df`` refers to the Titanic dataset.

```python
import numpy as np
import pandas as pd
df = pd.read_csv("/data301/data/titanic.csv")
df.head(2)  
```

## Describe each line of the following code segment and be specific when it comes to return types and behavior.

```python tags=["remove_output"]
%%capture
df = pd.read_csv("/data301/data/titanic.csv")
df.head()
```

$$\vspace{2in}$$


## Write the code that would set the index to ``name`` for the dataframe ``df``.


$$\vspace{2in}$$


## Assuming you have set the index to ``name`` in ``df``, what Pandas **type** would the following command return?
``df.loc["Allison, Master. Hudson Trevor"]``


$$\vspace{2in}$$


## What would the following return?
``
df.sex.describe()
``


$$\vspace{2in}$$


## Assuming you have set the index to ``name`` in ``df``, what would the following command return?
``df.fare.idxmax()``


$$\vspace{2in}$$


## What would happen if you tried to execute the following and why?
``df.iloc["Allison, Master. Hudson Trevor"]``


$$\vspace{2in}$$


## What are two different ways to return the ``age`` column without resorting to numerical indexing (i.e., no iloc)?


$$\vspace{2in}$$


## Write the code that will calculate the mean absolute deviation for the fare column.
$\textrm{MAD} = \textrm{mean of } |x_i - \bar x|$

$\noindent\textrm{MAD} = \frac{1}{n} \sum_{i=1}^n |x_i - \bar x|$


$$\vspace{2in}$$


## Given the titanic dataframe above, how could you use Pandas built-in string processing to split out the surname. Here is sample data.

```python
df.name.head()
```

$$\vspace{2in}$$


## Given the Ames dataset (previewed below), explain what the following code does and sketch the resulting graph.

```python
%matplotlib inline
import pandas as pd
pd.options.display.max_rows = 10

df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/AmesHousing.txt",
    sep="\t")
df.head(2) 
```

```python
df["Heating QC"].value_counts()
```

**Here is the code I want you to explain:**

```python
%%capture
df["Heating QC"].map({
       "Ex": "Excellent",
       "Gd": "Good",
       "TA": "Average",
       "Fa": "Fair",
       "Po": "Poor"
}).value_counts()[["Poor", "Fair", "Average", "Good", "Excellent"]].plot.bar()
```

$$\vspace{3in}$$


## Given the following picture, write the Pandas equivalent code if the original data on the left is stored in a dataframe called df. Indicate which part of your code corresponds to what step in the diagram.

![](split_apply_combine.png) [source](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.08-Aggregation-and-Grouping.ipynb)


$$\vspace{3in}$$

```latex
\newpage
```

## Consider the following code and output.

```python
df['adult'] = df['age']>=18
survivors_cube = df.pivot_table(
    index="sex", columns=["adult", "pclass"],
    values="survived", aggfunc=np.mean)
table = (survivors_cube*10).round().astype(int)
table
```

From the above data, what is the result of ``table.stack()``.


$$\vspace{2in}$$


## Given the following:

```python
adult_pclass_counts = pd.crosstab(df.pclass, df.adult)
adult_pclass_counts
```

```python
adult = adult_pclass_counts.sum(axis=0) / N
adult
```

```python
pclass = adult_pclass_counts.sum(axis=1) / N
pclass
```

```python
N = adult_pclass_counts.sum().sum()
joint = adult_pclass_counts / N
display(joint)
expected = np.outer(pclass, adult)
display(expected)
```

$$\noindent$$**What is the code for calculating Chi-square?**

$$\noindent$$**Chi-square distance** solves the problem of total variation distance by dividing by the difference by expected proportion, effectively calculating the _relative_ difference between the two proportions:

$$ \chi^2 = \sum_{\text{A, B}} \frac{(P(A \text{ and } B) - P(A) P(B))^2}{P(A) P(B)}. $$


$$\vspace{3in}$$


### What is the difference between covariance and correlation? Why would you want to look at one or the other?


$$\vspace{2in}$$


## (10 points) The table below is a sample from a dataset used to help determine if a house is acceptable to purchase. Calculate ``P(Acceptable=Yes|Not included,3,New)`` using the naive Bayes classifier described in class.

<!-- #region {"language": "latex"} -->
\begin{table}[!h]
    \centering
    \begin{tabular}{lllll}
    House & Furniture  & \# Rooms & Kitchen & Acceptable \\
    1     & Not included & 3         & New       & Yes          \\
    2     & Included     & 3         & Old       & No           \\
    3     & Not included & 4         & Old       & Yes          \\
    4     & Not included & 3         & New       & No           \\
    5     & Included     & 4         & Old       & Yes          \\
    \end{tabular}
\end{table}
<!-- #endregion -->

$$\vspace{4in}$$


## (10 points) Find the eigenvalues and eigenvectors of the following covariance matrix (circle answer). Show your work.

```latex
 \[
   \Sigma=
  \left[ {\begin{array}{cc}
   4 & 2 \\
   2 & 4 \\
  \end{array} } \right]
\]
```

$$\vspace{3in}$$


## (5 points) What is the percent variance explained in PC2 using your results from the last question?


$$\vspace{2in}$$

```python

```
