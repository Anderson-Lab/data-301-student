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

```python
import requests
```

```python
resp = requests.get("https://swapi.co/api/people/")
import json
data = resp.json()
 
names = [] 
eye_color = []
while True:
    for person in data['results']:
        names.append(person['name'])
        eye_color.append(person['eye_color'])
    if data["next"] is None:
        break
    resp = requests.get(data["next"])
    data = resp.json()
    
import pandas as pd
df = pd.DataFrame({"name":names,"eye_color":eye_color})
df.head()
```

```python
data
```

```python
import requests
from pandas.io.json import json_normalize

resp = requests.get("https://swapi.co/api/people/")
data = resp.json()

df = None
while True:
    current_df = json_normalize(data['results'])
    if df is None:
        df = current_df
    else:
        df = df.append(current_df)

    if data["next"] is None:
        break
    resp = requests.get(data["next"])
    data = resp.json()
```

```python
df.groupby("homeworld").eye_color.value_counts()
```

```python
import requests
from pandas.io.json import json_normalize

def get_all_info(key,arg1=None,meta=None):
    resp = requests.get("https://swapi.co/api/%s/"%key)
    data = resp.json()

    df = None
    while True:
        current_df = json_normalize(data['results'],arg1,meta=meta)
        if df is None:
            df = current_df
        else:
            df = df.append(current_df)

        if data["next"] is None:
            break
        resp = requests.get(data["next"])
        data = resp.json()
    return df
```

```python
people = get_all_info("people","films",meta=["name","eye_color"])
```

```python
people.head()
```

```python
planets= get_all_info("planets")
```

```python
planets.head()
```

```python
people_planets = people.set_index("homeworld").join(planets.set_index("url"),lsuffix="_people",rsuffix="_planets")
```

```python
people_planets.index.name = "homeworld"
people_planets = people_planets.reset_index()
```

```python
people_planets.groupby("name_planets").eye_color.value_counts().loc["Tatooine"]
```

```python
people_planets.loc[people_planets.name_planets=="Tatooine"]
```

```python
films= get_all_info("films")
```

```python
films
```

```python
df = films.set_index("url").join(people.set_index(0))
```

```python
df.head()
```

```python
df.pivot_table(index="title",columns="eye_color",values="name",aggfunc='count')
```

```python

```
