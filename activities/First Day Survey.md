---
jupyter:
  jupytext:
    formats: ipynb,py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# First-Day Survey

First, fill out the [first-day survey](https://goo.gl/forms/MRg5GpHvd6ECMcav1), which is an anonymous survey with some fun questions. This notebook will read in [the responses](https://docs.google.com/spreadsheets/d/1NzcMTL7INHHee8CDdcSPPgyKuiEg0YTkqoOkJbI9JmQ/) into a `pandas` `DataFrame`.

We'll do this by issuing a request to the [Google Sheets REST API](https://developers.google.com/sheets/api/reference/rest/), which returns the data in JSON format. Don't worry if all of this is Greek to you now. You'll learn what all of this means, later in the course. For now, just run the cell below and trust that it fetches the data in the spreadsheet above.

```python
import requests

API_KEY = "AIzaSyAu1_itQekOyIXrIKIfn9sJrLVVGCL3Unc"
SPREADSHEET_ID = "1NzcMTL7INHHee8CDdcSPPgyKuiEg0YTkqoOkJbI9JmQ"

url = "https://sheets.googleapis.com/v4/spreadsheets/%s/values/A1:I100?key=%s" % (
    SPREADSHEET_ID,
    API_KEY
)
req = requests.get(url)
```

All of the survey responses are in the `values` attribute of the resulting JSON object. Let's extract that and construct a `pandas` `DataFrame` out of it.

```python
import pandas as pd
df = pd.DataFrame(req.json()["values"])
df.head()
```

There's a problem. The first row (i.e., row 0) is clearly supposed to be the column names. To make things right, we can rename the columns to the values in row 0 and drop row 0 from the `DataFrame`.

```python
df.rename(columns=df.iloc[0]).drop(0)
```

Now save the `DataFrame` above into a variable, and start exploring this data set!

```python
# YOUR CODE HERE

```
