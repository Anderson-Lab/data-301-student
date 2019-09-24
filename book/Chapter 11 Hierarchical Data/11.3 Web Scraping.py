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

# # 11.3 Web Scraping
#
# **HTML**, which stands for "hypertext markup language", is an XML-like language for specifying the appearance of web pages. Each tag in HTML corresponds to a specific page element. For example:
#
# - `<img>` specifies an image. The path to the image file is specified in the `src=` attribute.
# - `<a>` specifies a hyperlink. The text enclosed between `<a>` and `</a>` is the text of the link that appears, while the URL is specified in the `href=` attribute of the tag.
# - `<table>` specifies a table. The rows of the table are specified by `<tr>` tags nested inside the `<table>` tag, while the cells in each row are specified by `<td>` tags nested inside each `<tr>` tag.
#
# Our goal in this section is not to teach you HTML to make a web page. You will learn just enough HTML to be able to scrape data programmatically from a web page.

# # Inspecting HTML Source Code
#
# Suppose we want to scrape faculty information from the [Cal Poly Statistics Department directory](https://statistics.calpoly.edu/content/StatisticsDirectory%26Office%20Hours) (`https://statistics.calpoly.edu/content/StatisticsDirectory%26Office%20Hours`). Once we have identified a web page that we want to scrape, the next step is to study the HTML source code. All web browsers have a "View Source" or "Page Source" feature that will display the HTML source of a web page. 
#
# Visit the web page above, and view the HTML source of that page. (You may have to search online to figure out how to view the page source in your favorite browser.) Scroll down until you find the HTML code for the table containing information about the name, office, phone, e-mail, and office hours of the faculty members.
#
# Notice how difficult it can be to find a page element in the HTML source. Many browsers allow you to right-click on a page element and jump to the part of the HTML source corresponding to that element.

# # Web Scraping Using BeautifulSoup
#
# `BeautifulSoup` is a Python library that makes it easy to navigate an HTML document. Like with `lxml`, we can query tags by name or attribute, and we can narrow our search to the ancestors and descendants of specific tags. In fact, it is possible to use `lxml` with HTML documents, but many web sites have malformed HTML, and `lxml` is not very forgiving. `BeautifulSoup` handles malformed HTML more gracefully and is thus the library of choice.
#
# First, we issue an HTTP request to the URL to get the HTML source code.

import requests
resp = requests.get("https://statistics.calpoly.edu/content/StatisticsDirectory%26Office%20Hours")

# The HTML source is stored in the `.content` attribute of the response object. We pass this HTML source into `BeautifulSoup` to obtain a tree-like representation of the HTML document.

from bs4 import BeautifulSoup
soup = BeautifulSoup(resp.content, "html.parser")

# Now we can search for tags within this HTML document, using tags like `.find_all()`. For example, we can find all tables on this page.

tables = soup.find_all("table")
len(tables)

# As a visual inspection of [the web page](https://statistics.calpoly.edu/content/StatisticsDirectory%26Office%20Hours) would confirm, there are 2 tables on the page, and we are interested in the second one.

table = tables[1]
table

# There is one faculty member per row, except for the first row, which is the header. We iterate over all rows except for the first, extracting information about each faculty to append to `rows`, which we will eventually turn into a `DataFrame`. As you read the code below, refer to the HTML source above, so that you understand what each line is doing. In particular, ask yourself the following questions:
#
# - `cells[0]` represents a `<td>` tag. Why do we need to call `.find("strong")` within this tag to get the name of the faculty member?
# - For the most part, `link` is a hyperlink whose text is the faculty's office number. But for some faculty, `link` is `None`. For which faculty is this the case and why?
#
# You are encouraged to add `print()` statements inside the `for` loop to check your understanding of each line of code.

rows = []
for faculty in table.find_all("tr")[1:]:
    # Get all the cells in the row.
    cells = faculty.find_all("td")
    
    # The information we need is the text between tags.
    name = cells[0].find("strong").text
    
    link = cells[1].find("a")
    office = cells[1].text if link is None else link.text
    
    email = cells[3].find("a").text
    
    # Append this data.
    rows.append({
        "name": name,
        "office": office,
        "email": email
    })

# In the code above, observe that `.find_all()` returns a list with all matching tags, while `.find()` returns only the first matching tag. If no matching tags are found, then `.find_all()` will return an empty list `[]`, while `.find()` will return `None`.
#
# Finally, we turn `rows` into a `DataFrame`.

import pandas as pd
pd.DataFrame(rows)

# Now this data is ready for further processing.

# # Ethical Enlightenment: `robots.txt`
#
# Web robots are crawling the web all the time. A website may want to restrict the robots from crawling specific pages. One reason is financial: each visit to a web page, by a human or a robot, costs the website money, and the website may prefer to save their limited bandwidth for human visitors. Another reason is privacy: a website may not want a search engine to preserve a snapshot of a page for all eternity.
#
# To specify what a web robot is and isn't allowed to crawl, most websites will place a text file named `robots.txt` in the top-level directory of the web server. For example, the Statistics department web page has a `robots.txt` file: https://statistics.calpoly.edu/robots.txt
#
# The format of the `robots.txt` file should be self-explanatory, but you can read a full specification of the standard here: http://www.robotstxt.org/robotstxt.html. As you scrape websites using your web robot, always check the `robots.txt` file first, to make sure that you are respecting the wishes of the website owner.

# # Exercises

# **Exercise 1.** The [Cal Poly course catalog](http://catalog.calpoly.edu/collegesandprograms/collegeofsciencemathematics/statistics/#courseinventory) (`http://catalog.calpoly.edu/collegesandprograms/collegeofsciencemathematics/statistics/#courseinventory`) contains a list of courses offered by the Statistics department. Scrape this website to obtain a `DataFrame`, where the unit of observation is a course. Use this `DataFrame` to answer the following questions: 
#
# - How many 300-level courses does the Statistics department offer?
# - How many distinct courses are offered in Spring quarter?

# +
# YOUR CODE HERE
