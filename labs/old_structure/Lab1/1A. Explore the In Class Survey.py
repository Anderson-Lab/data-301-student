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

# # Explore the In Class Survey
#
# During the first class, you filled out the [first-day survey](https://goo.gl/forms/MRg5GpHvd6ECMcav1). In this lab, you will explore [the responses](https://docs.google.com/spreadsheets/d/1NzcMTL7INHHee8CDdcSPPgyKuiEg0YTkqoOkJbI9JmQ/).
#
# Run the following code to read the data into a `pandas` `DataFrame` whose columns are the survey questions. Each row represents one student's response to the questions.

# +
import pandas as pd
import requests

API_KEY = "AIzaSyAu1_itQekOyIXrIKIfn9sJrLVVGCL3Unc"
SPREADSHEET_ID = "1NzcMTL7INHHee8CDdcSPPgyKuiEg0YTkqoOkJbI9JmQ"

url = "https://sheets.googleapis.com/v4/spreadsheets/%s/values/A1:J100?key=%s" % (
    SPREADSHEET_ID,
    API_KEY
)
req = requests.get(url)
df = pd.DataFrame(req.json()["values"])
df = df.rename(columns=df.iloc[0]).drop(0)

df.head()
# -

# ## Question 1
#
# Calculate the number of siblings (total, both older and younger) each student has. Make a graphic that visualizes this information. Explain what you see.

# +
# TYPE YOUR CODE HERE.
# -

# **TYPE YOUR WRITTEN EXPLANATION HERE.**

# ## Question 2
#
# Make a graphic that visualizes the favorite colors of students in DATA 301. Explain what you see.
#
# (_Hint:_ You might have to clean the data a bit first.)

# +
# TYPE YOUR CODE HERE.
# -

# **TYPE YOUR WRITTEN EXPLANATION HERE.**

# ## Question 3
#
# Remember that wacky question about how many basketballs would fit in the classroom? Unbeknownst to you, I actually presented the question differently to the two sections.
#
# - The morning section was first asked, "Do you think that we would need more or less than 1,000 basketballs?"
# - The afternoon section was first asked "Do you think that we would need more or less than 100,000 basketballs?"
#
# The exact number that each student was given in the prompt is stored in the "Prompt" column of the `DataFrame`.
#
# The purpose of this exercise was to test a famous effect in psychology called the ["anchoring effect"](https://en.wikipedia.org/wiki/Heuristics_in_judgment_and_decision-making#Anchoring_and_adjustment). The hypothesis is that the afternoon section, which was presented with the higher "anchor", would guess larger numbers than the morning section.
#
# Does the data provide evidence of an anchoring effect? Explain your approach and state your conclusions.
#
# (_Hint:_ There are many reasonable approaches to this problem. You will get full credit for any reasonable approach, as long as you carefully justify it.)

# +
# TYPE YOUR CODE HERE.
# -

# **TYPE YOUR WRITTEN EXPLANATION HERE.**

# ## Submission Instructions
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
# 3. Upload the PDF [to PolyLearn](https://polylearn.calpoly.edu/AY_2018-2019/mod/assign/view.php?id=296993).
