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

# # 11.2 The XML Data Format
#
# **XML**, which stands for eXtensible Markup Language, is another way to represent hierarchical data. The basic building block of XML is the **tag**, denoted by angle brackets `<>`.
#
# For example, a data set of movies might be represented using XML as follows:
#
# ```
# <movies>
#   <movie id="1" title="The Godfather">
#     <director id="50" name="Coppola, Francis Ford">
#     </director>
#     <releasedate>1972-03-24</releasedate>
#     <character id="100" name="Vito Corleone">
#       <actor id="200" name="Brando, Marlon">
#       </actor>
#     </character>
#     <character id="101" name="Michael Corleone">
#       <actor id="201" name="Pacino, Al">
#       </actor>
#     </character>
#     ...
#   </movie>
#   <movie id="2" title="The Godfather: Part II">
#     <director id="50" name="Coppola, Francis Ford">
#     </director>
#     <releasedate>1974-10-20</releasedate>
#     <character id="101" name="Michael Corleone">
#       <actor id="201" name="Pacino, Al">
#       </actor>
#     </character>
#     <character id="100" name="Vito Corleone">
#       <actor id="250" name="De Niro, Robert">
#       </actor>
#     </character>
#     ...
#   </movie>
#   ...
# </movies>
# ```
#
# Note the following features of XML:
#
# - Every tag `<a>` has a corresponding closing tag `</a>`. You can always recognize a closing tag by the forward slash `/`.
# - Additional tags and/or strings can be nested between the opening and closing tags. In the example above, `<actor>` is nested between `<character>` and `</character>`, and `<character>` is nested between `<movie>` and `</movie>`. The nesting is used to represent hierarchy.
# - Indentation is used to make the code more readable (to make it easier to see the nesting structure). But it is optional.
# - Attributes can be associated with each tag, like `id=` and `name=` with the `<character>` tag and `id=` and `title=` with the `<movie>` tag.
#
# Each tag represents a variable in the data set. Unlike JSON, which uses lists to represent repeated fields, XML represents repeated fields by simply repeating tags where necessary. In the example above, there are multiple instances of `<movie>` within `<movies>` and multiple instances of `<character>` within `<movie>`, so `movie` and `character` are both repeated fields. (In fact, `director` is also a repeated field, but it is impossible to tell from the code above, since the movies shown above only have one director.)
#
# You will learn XML by working with the same New York Philharmonic data as in the previous section, except that the data is now stored in XML format. Let's look at this file on disk.

# !ls -l /data301/data/nyphil/

# Notice that this XML file is nearly twice as large as the JSON file. Although XML is more readable than JSON, it is a more expensive way to store hierarchical data, primarily because of the cost of storing both the opening and closing tags.

# There are several libraries in Python for working with XML, including BeautifulSoup (which we will use in the next section to parse HTML), ElementTree, and `lxml`. We will use `lxml` to work with XML data because it is fastest for large data sets, provided that the data is well-formed. The `lxml` library provides a convenient API that replicates all of the functionality of ElementTree, plus implements a few additional features that are useful for data analysis.

from lxml import etree

# First, let's read in the data using `lxml`. The `.parse()` function of ElementTree reads in an XML document from a file or URL and returns a tree-like data structure called an ElementTree.

# !head /data301/data/nyphil/complete.xml

tree = etree.parse("/data301/data/nyphil/complete.xml")

# Every XML document has a single "root" tag that encloses all of the other tags. For the New York Philharmonic data, this root tag is `<programs>`.

tree.getroot()

# If the XML data is already stored as a string in memory, then we instead use the `.fromstring()` method. Note that `.fromstring()` returns the root tag directly.

# +
with open("/data301/data/nyphil/complete.xml", "rb") as f:
    string = f.read()
    
etree.fromstring(string)
# -

# Each direct descendant, or **child**, of `<programs>` is a program. To find the direct descendants of a tag, we call the `.getchildren()` method.

programs = tree.getroot()
print(len(programs.getchildren()))
programs.getchildren()[:10]

# Let's print out the first of these programs. There are two ways to get the first program.

# +
# METHOD 1: Get it from the list above.
program = programs.getchildren()[0]

# METHOD 2: Use .find() to find the first instance of a tag.
program = tree.find("program")
program
# -

# Now let's see how the data is represented by printing out the XML of this program. To do this, we use the `etree.tostring()` function.

print(etree.tostring(program, encoding="unicode"))

# Hopefully, the basic structure of this data is already familiar to you from previous section. "Work", "concertInfo", and "soloist" are repeated fields inside "program". One difference between the JSON and the XML is that "work" is not directly nested within "program"; the "work" tags are all nested inside an additional "worksInfo" tag.
#
# Now suppose that we want to flatten the data at the level of soloists. To get all of the soloists, we can use the `.findall()` method. Let's first try the obvious solution, which does not work:

programs.findall("soloist")

# Why did `lxml` fail to find any `<soloist>` tags? That's because `.findall()` only searches among the direct descendants of a tag. We called `.findall()` on the `<programs>` tag, but all of its descendants are `<program>` tags.
#
# To specify that `lxml` should look for `<soloist>` tags among all descendants, not just direct ones, we use the `.xpath()` command, which allows us to specify an XPath expression. [XPath](https://www.w3schools.com/xml/xpath_syntax.asp) is a language used to select nodes from XML documents. The XPath expression to select all descendants named `<soloist>` of the current tag is `".//soloist"`. We pass this expression to the `.xpath()` method.

soloists = programs.xpath(".//soloist")
len(soloists)

# Now, to flatten the data at the level of soloists, we just need to turn `soloists` into a `DataFrame` with as many rows. But what if we want to include information from parent levels, like the composer of the work the soloist played? There are two ways.
#
# ### Method 1
#
# Since `<composerName>` is a descendant of `<work>`, one way is to navigate up to the level of `<work>` by calling `.getparent()` repeatedly and then find `<composerName>` among its descendants:

# +
soloist = soloists[0]

# The first .getparent() returns the <soloists> tag.
# The second .getparent() returns the <work> tag.
# You have to figure this out by inspecting the XML.
work = soloist.getparent().getparent()
work.xpath(".//composerName")
# -

# This is a list with one tag, so we extract that tag and the text inside it.

work.xpath(".//composerName")[0].text

# ### Method 2
#
# As the number of levels of nesting increases, it quickly becomes impractical to call `.getparent()` repeatedly. We want to be able to jump directly to the right ancestor. The easiest way to do this is to use the XPath expression for an ancestor. To search for all ancestors named "work", we can use the XPath expression `"ancestor::work"`.

soloist.xpath("ancestor::work")

# Now, we can extract this single work tag and find its descendants named `<composerName>`. Or better yet, we can combine this step with the above step into a single XPath expression.

soloist.xpath("ancestor::work//composerName")[0].text

# Now let's put it all together. We will flatten the data to get a `DataFrame` with one soloist per row. We will keep track of the soloist's name, instrument, and role---as well as the composer of the work they performed. Unfortunately, it is much more manual to do this with XML than with JSON. There is no XML equivalent of the `json_normalize` function that will automatically produce a `DataFrame`, so we have to construct the `DataFrame` ourselves.

# +
import pandas as pd

rows = []

soloists = programs.xpath(".//soloist")
for soloist in soloists:
    row = {}
    row["soloistName"] = soloist.find("soloistName").text
    row["soloistInstrument"] = soloist.find("soloistInstrument").text
    row["soloistRoles"] = soloist.find("soloistRoles").text
    row["composerName"] = soloist.xpath("ancestor::work//composerName")[0].text
    rows.append(row)
    
print(rows[:5])
    
soloistsdf = pd.DataFrame(rows)
soloistsdf
# -

# Now, this is a `DataFrame` that we can analyze easily. For example, here is how many times Benny Goodman programmed a work by Mozart with the NY Phil:

soloistsdf[soloistsdf["soloistName"] == "Goodman, Benny"].composerName.value_counts()

# # RESTful Web Services
#
# Many RESTful web services return data in XML format. Like before, we use the `requests` library in Python to issue the HTTP request. For example, the website [FloatRates](http://www.floatrates.com/feeds.html) provides exchange rates between world currencies in XML format.

import requests
resp = requests.get("http://www.floatrates.com/daily/usd.xml")
resp

# The XML is stored in the `.content` attribute of the response object. We can parse this string into an ElementTree using the `.fromstring()` function in the `lxml` library. Recall that this returns the root tag of the XML document.

etree.fromstring(resp.content)

# # Exercises
#
# Exercises 1 and 2 deal with the New York Philharmonic data set from above. These exercises are the same as the ones in the previous section, except that now you have to do them with XML.

# **Exercise 1.** What is the most frequent start time for New York Philharmonic concerts?

# +
# TYPE YOUR CODE HERE
# BEGIN SOLUTION
from lxml import etree
import pandas as pd

tree = etree.parse("/data301/data/nyphil/complete.xml")

concerts = tree.xpath("/programs/program/concertInfo")
times = []
for concert in concerts:
    times.append(concert.find("Time").text)
df = pd.DataFrame({"time":times})
df["time"].value_counts()
# END SOLUTION
# -

programs

# **Exercise 2.** How many total concerts did the New York Philharmonic perform in the 2014-15 season?

# +
# TYPE YOUR CODE HERE
# BEGIN SOLUTION
from lxml import etree
import pandas as pd

tree = etree.parse("/data301/data/nyphil/complete.xml")

seasons = tree.xpath("/programs/program/season")
c = 0
for season in seasons:
    if season.text == "2014-15":
        c+=1
print(c)
print('This does not match up, but I am not sure why at the moment. If they have something that pulls out the season, then it is good in my books')
# END SOLUTION
# -


