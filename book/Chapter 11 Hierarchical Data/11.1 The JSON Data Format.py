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

# # Chapter 11. Hierarchical Data
#
# A lot of data in the real world is naturally hierarchical. For example, consider a data set of concert programs by the New York Philharmonic, one of the world's leading orchestras. Each program consists of one or more works of music and is performed at one or more concerts. Furthermore, each work of music may feature any number of soloists.
#
# How would we represent this information in a single `DataFrame`? If each row represents a single program, then we need one column for each concert that the program appeared in. This is wasteful because some programs may have only appeared in one concert. We still need to keep around $M$ "concert" columns, where $M$ is the maximum number of concerts that any program appeared in.
#
# |concert1    | concert2   | ... | concertM | work1 | work2 | ... | workN |
# |------------|------------|-----|----------|----------------|-------|-----|-------|
# | 2016-12-11 | `NaN`      | ... | `NaN`    | Violin Concerto No. 2 | Symphony No. 5  | ... | `NaN` |
# | 2016-12-13 | 2016-12-14 | ... | 2016-12-17 | Messiah | `NaN` | ... | `NaN` |
# | ... | ... | ... | ... | ... | ... | ... | ... |
#
# Similarly, we need one column for each work in the program. The number of "work" columns has to be equal to the maximum number of works on any program, even though most programs may have had far fewer works. 
#
# Hopefully, it is clear that a single `DataFrame` is an inefficient way to represent hierarchical data---and we haven't even tried to include information about the soloists who performed in each work. This chapter is about efficient ways to represent hierarchical data, like the New York Philharmonic data set described above.

# # Chapter 11.1 The JSON Data Format
#
# The JavaScript Object Notation, or **JSON**, data format is a popular way to represent hierarchical data. Despite its name, its application extends far beyond JavaScript, the language for which it was originally designed.
#
# Let's take a look at the first 1000 characters of a JSON file. (_Warning:_ Never try to print the entire contents of a JSON file in a Jupyter notebook; this will freeze the notebook if the file is large!)

# !head -c 1000 /data301/data/nyphil/complete.json

# Hopefully, this notation is familiar. It is just the notation for a Python dictionary! Although there are a few cosmetic differences between Python dicts and JSON, they are the same for the most part, and we will use the terms "dict" and "JSON" interchangeably. 
#
# The `json` library in Python allows you to read a JSON file directly into a Python dict.

import json
with open("/data301/data/nyphil/complete.json") as f:
    nyphil = json.load(f)

# Let's take a look at this Python dict that we just created, again being careful not to print out the entire dict. Let's just take a look at the first two programs in the data set. This should hopefully be enough to give you a sense of how the data is structured.

nyphil["programs"][:2]

# The top-level variables in each "program" are:
#
# - concerts
# - id
# - orchestra
# - programID
# - season
# - works
#
# Most of these variables are fairly standard; the only interesting ones are "concerts" and "works", which are both lists. A variable that is a list is called a **repeated field**. A repeated field might itself consist of several variables (for example, each "work" has a composer, a conductor, and soloists), thus creating a hierarchy of variables. Repeated fields are what makes a data set hierarchical.

# # Flattening Hierarchical Data
#
# How many distinct works by Ludwig van Beethoven has the New York Philharmonic performed? Answering this question from the Python dict is irritating, as it involves writing multiple nested "for" loops to traverse the JSON data. Shown below is the code to do this, although we will see an easier way shortly.

# +
# Spaghetti Code (Don't do this --- see below for an easier way.)
beethoven = set()
for program in nyphil["programs"]:
    for work in program["works"]:
        if "composerName" in work and work["composerName"] == "Beethoven,  Ludwig  van":
            beethoven.add(work["workTitle"])
            
len(beethoven)
# -

# The only data that we really need to answer the above question is a `DataFrame` of works that the New York Philharmonic has performed. To obtain such a `DataFrame`, we need to **flatten** the JSON data at the level of "work" to produce a `DataFrame` with one row per work. The `json_normalize()` in `pandas.io.json` is a function that allows us to flatten JSON data at any desired level. The first argument to `json_normalize()` is the JSON data (i.e., a Python dict), and the second argument specifies the level at which to flatten.

# +
import pandas as pd
from pandas.io.json import json_normalize
pd.options.display.max_rows = 10

works = json_normalize(nyphil["programs"], "works")
display(works)
display(nyphil["programs"][0]["works"][0])
display(works.columns)
# -

# Note that this flattening operation resulted in some loss of information. We no longer have information about the program that each work appeared in. We can partly alleviate this problem by specifying "metadata" from parent levels to append. For example, "season" and "orchestra" are properties of "program", which is the parent of "work". If we want to include these variables with each work, then we pass them to the `meta=` argument of `json_normalize()`.

json_normalize(nyphil["programs"], "works", meta=["season", "orchestra"])

# However, there is still some loss of information. For example, there is no way to tell from this flattened `DataFrame` which works appeared together on the same program. (In the case of this particular data set, there is a "programID" that could be used to preserve information about the program, but not all data sets will have such an ID.)
#
# Note also that repeated fields that are nested within "work", such as "soloist", remain unflattened. They simply remain as a list of JSON objects embedded within the `DataFrame`. They are not particularly accessible to analysis.
#
# But now that we have a `DataFrame` with one row per work, we can determine the number of unique Beethoven works that the Philharmonic has performed by subsetting the `DataFrame` and grouping by the title of the work.

beethoven = works[works.composerName == "Beethoven,  Ludwig  van"]
len(beethoven.groupby("workTitle")["ID"].count())

# What if we wanted to know how many works Benny Goodman has performed with the New York Philharmonic? We could flatten the data at the level of the "soloist". Since "soloists" is nested within "works", we specify a path (i.e., `["works", "soloists"]`) as the flattening level.

soloists = json_normalize(nyphil["programs"], ["works", "soloists"])
soloists

# Now we can use this flattened `DataFrame` to easily answer the question.

(soloists["soloistName"] == "Goodman, Benny").sum()

# If we wanted to know how many works by Mozart that Goodman performed, we need to additionally store the "composerName" from the "works" level. We do this by specifying the path to "composerName" (i.e., `["works", "soloists"]`) in the `meta=` argument. But there is a catch. There are some works where the "composerName" field is missing. `json_normalize()` will fail if it cannot find the "composerName" key for even a single work. So we have to manually go through the JSON object and manually add "composerName" to the object, setting its value to `None`, if it does not exist.

for program in nyphil["programs"]:
    for work in program["works"]:
        if "composerName" not in work:
            work["composerName"] = None

soloists = json_normalize(
    nyphil["programs"],
    ["works", "soloists"], 
    meta=[["works", "composerName"], "season"]
)
soloists

soloists[soloists["soloistName"] == "Goodman, Benny"]["works.composerName"].value_counts()

# # RESTful Web Services
#
# One way that organizations expose their data to the public is through RESTful web services. In a typical RESTful service, the user specifies the kind of data they want in the URL, and the server returns the desired data. JSON is a common format for returning data.
#
# For example, the [Star Wars API](http://swapi.co) is a RESTful web service that returns data about the Star Wars universe, including characters, spaceships, and planets. To look up information about characters named "Skywalker", we would issue an HTTP request to the URL http://swapi.co/api/people/?search=skywalker. Notice that this returns data in JSON format.
#
# To issue the HTTP request within Python (so that we can further process the JSON), we can use the `requests` library in Python.

import requests
resp = requests.get("http://swapi.co/api/people/?search=skywalker")
resp

# The response object contains the JSON and other metadata. To extract the JSON in the form of a Python dict, we call `.json()` on the response object.

skywalker = resp.json()
skywalker

from pandas.io.json import json_normalize

# Now we can process this data just like we did with the JSON data that we read in from a file.

json_normalize(skywalker, "results")

# # Ethical Enlightenment: Staggering Requests
#
# Suppose you want information about the starships associated with the Skywalkers you found above. If we flatten the JSON object at the "starships" level, then we get a list of URLs that we can query to get information about each starship.

starship_urls = json_normalize(skywalker, ["results", "starships"])
starship_urls

# It is straightforward enough to write a loop that queries each of these URLs and saves the corresponding JSON object. However, a script can easily issue hundreds, even thousands, of queries per second, and we want to avoid spamming the server. (In fact, if a website detects many requests coming from the same IP address, it may think it is being attacked and block the IP address.)
#
# To respect the host, who is providing this information for free, we stagger the queries by inserting a delay. This can be done using `time.sleep()`, which will suspend execution of the script for the given number of seconds. We will add a half second delay (so that we make no more than 2 queries per second) between requests.

# +
import time

starships = []
for starship_url in starship_urls[0]:
    
    # get the JSON for the starship from the REST API
    resp = requests.get(starship_url)
    starships.append(resp.json())
    
    # add a 0.5 second delay between each query
    time.sleep(0.5)
    
starships
# -

# # Exercises
#
# Exercises 1-3 deal with the New York Philharmonic data set from above.

# **Exercise 1.** Answer the Benny Goodman question above ("How many works has Benny Goodman performed with the New York Philharmonic?") by writing nested for loops that traverse the structure of the JSON object. Check that your answer agrees with the one we obtained above by first flattening the JSON object to a `DataFrame`.

# +
# ENTER YOUR CODE HERE.
# BEGIN SOLUTION
import json
with open("/data301/data/nyphil/complete.json") as f:
    nyphil = json.load(f)
    
c = 0
for i in range(len(nyphil['programs'])):
    for j in range(len(nyphil['programs'][i]['works'])):
        for k in range(len(nyphil['programs'][i]['works'][j]['soloists'])):
            if nyphil['programs'][i]['works'][j]['soloists'][k]['soloistName'] == "Goodman, Benny":
                c+=1
print(c)
# END SOLUTION
# -

# **Exercise 2.** What is the most frequent start time for New York Philharmonic concerts?

# ENTER YOUR CODE HERE.
# BEGIN SOLUTION
from pandas.io.json import json_normalize
concerts = json_normalize(nyphil["programs"], "concerts")
concerts.Time.value_counts()
# END SOLUTION

# **Exercise 3.** How many total concerts did the New York Philharmonic perform in the 2014-15 season?

# ENTER YOUR CODE HERE.
# BEGIN SOLUTION
from pandas.io.json import json_normalize
concerts = json_normalize(nyphil["programs"], "concerts",meta=['season'])
concerts.pivot_table(index='season',columns='Time',aggfunc='count').loc['2014-15'].sum()
# END SOLUTION

# To answer Exercises 4-6, you will need to issue HTTP requests to the Open States API, which contains information about state legislatures. You will need to include an API key with every request. You can [register for an API key here](https://openstates.org/api/register/). Once you have an API key, enter your API key below. If your API key works, then the code below should produce a `DataFrame` of all of the committees in the California State Assembly (the lower chamber).

# # I'm making this extra credit. If they can make decent progress on any of them, then it's +1 to their score. The API is just too damn annoying.

# +
# This is just a sample request to test that your API key is working.
import requests
import pandas as pd

apikey = ""
# BEGIN SOLUTION
apikey = "27fbd87e-a817-45f5-a844-7768daffba2d"
# END SOLUTION
headers = {
    'X-API-KEY': apikey
}
resp = requests.get(
    "https://openstates.org/api/v1/legislators/?state=nc&chamber=lower", headers=headers
)
pd.DataFrame(resp.json())
#pd.DataFrame(resp.json())
# -

# To answer the questions below, you will need to issue your own HTTP requests to the API. To understand how to construct URLs, you will need to refer to the [documentation for this API](http://docs.openstates.org/en/latest/api/).

# **Exercise 4.** Legislators typically have offices in both the Capitol building and in their districts. Among the active legislators in the California Assembly (lower chamber), which legislators have the most offices (and how many do they have)?

# ENTER YOUR CODE HERE.
# BEGIN SOLUTION
apikey = "27fbd87e-a817-45f5-a844-7768daffba2d"
# END SOLUTION
headers = {
    'X-API-KEY': apikey
}
resp = requests.get(
    "https://openstates.org/api/v1/legislators/?state=ca&chamber=lower", headers=headers
)
offices = json_normalize(resp.json(), "offices", meta=['active','full_name'])
pivot_df = offices.pivot_table(index="full_name",columns="active",aggfunc='count',values='address')
display(pivot_df.iloc[:,0].sort_values(ascending=False))
# END SOLUTION

df['offices']

# **Exercise 5.** Get all of the _constitutional amendments_ in the California State Senate (upper house) from the current legislative session. How many amendments have there been?
#
# (_Hint:_ "Constitutional amendment" is a type of bill.)

# +
# ENTER YOUR CODE HERE.
# BEGIN SOLUTION
# This is just a sample request to test that your API key is working.
import requests
import pandas as pd

apikey = ""
# BEGIN SOLUTION
apikey = "27fbd87e-a817-45f5-a844-7768daffba2d"
# END SOLUTION
headers = {
    'X-API-KEY': apikey
}

pages = []
c=1
max_pages = 10000
while True:
    resp = requests.get(
        "https://openstates.org/api/v1/bills/?state=nd&page=%d&chamber=upper&session=20182019"%c, headers=headers
    )

    pages.extend(resp.json())
    c+=1
    print(c)
    if c > max_pages:
        break

# END SOLUTION
# -

pages[:1]

# BEGIN SOLUTION
json_normalize(pages,'type')[0].value_counts()
#json_normalize(pages)
# END SOLUTION

# **Exercise 6.** Look up the votes on the constitutional amendments you found in Exercise 5. Calculate the number of "yes" and "no" votes for each legislator on these amendments. Which legislator had the most total votes on constitutional amendments in the current session? Which legislator had the most total negative votes?

# ENTER YOUR CODE HERE.
# BEGIN SOLUTION
df = json_normalize(pages,'type',meta=['id'])
df.columns = ["type","id","votes"]
ca_df = df.loc[df.type == 'constitutional amendment',:]
ca_df
# END SOLUTION

df = json_normalize(pages)
df

json_normalize(pages).columns#,'votes')

# BEGIN SOLUTION
apikey = "27fbd87e-a817-45f5-a844-7768daffba2d"
# END SOLUTION
headers = {
    'X-API-KEY': apikey
}
resp = requests.get(
    "https://openstates.org/api/v1/legislators/?state=ca&chamber=lower", headers=headers
)
json_normalize(resp.json()).columns


