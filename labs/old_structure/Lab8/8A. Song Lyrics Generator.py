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

# # 8A. Song Lyrics Generator
#
# In this lab, you will scrape a website to get lyrics of songs by your favorite artist. Then, you will train a model called a Markov chain on these lyrics so that you can generate a song in the style of your favorite artist.
#
# # Question 1. Scraping Song Lyrics
#
# Find a web site that has lyrics for several songs by your favorite artist. Scrape the lyrics into a Python list called `lyrics`, where each element of the list represents the lyrics of one song.
#
# **Tips:**
# - Find a web page that has links to all of the songs, like [this one](http://www.azlyrics.com/n/nirvana.html). [_Note:_ It appears that `azlyrics.com` blocks web scraping, so you'll have to find a different lyrics web site.] Then, you can scrape this page, extract the hyperlinks, and issue new HTTP requests to each hyperlink to get each song. 
# - Use `time.sleep()` to stagger your HTTP requests so that you do not get banned by the website for making too many requests.

# +
import requests
import time

from bs4 import BeautifulSoup

# +
lyrics = []

# YOUR CODE HERE
# -

# Print out the lyrics to the first song.
print(lyrics[0])

# `pickle` is a Python library that serializes Python objects to disk so that you can load them in later.

import pickle
pickle.dump(lyrics, open("lyrics.pkl", "wb"))


# # Question 2. Unigram Markov Chain Model
#
# You will build a Markov chain for the artist whose lyrics you scraped in Lab A. Your model will process the lyrics and store the word transitions for that artist. The transitions will be stored in a dict called `chain`, which maps each word to a list of "next" words.
#
# For example, if your song was ["The Joker" by the Steve Miller Band](https://www.youtube.com/watch?v=FgDU17xqNXo), `chain` might look as follows:
#
# ```
# chain = {
#     "some": ["people", "call", "people"],
#     "call": ["me", "me", "me"],
#     "the": ["space", "gangster", "pompitous", ...],
#     "me": ["the", "the", "Maurice"],
#     ...
# }
# ```
#
# Besides words, you should include a few additional states in your Markov chain. You should have `"<START>"` and `"<END>"` states so that we can keep track of how songs are likely to begin and end. You should also include a state called `"<N>"` to denote line breaks so that you can keep track of where lines begin and end. It is up to you whether you want to include normalize case and strip punctuation.
#
# So for example, for ["The Joker"](https://www.azlyrics.com/lyrics/stevemillerband/thejoker.html), you would add the following to your chain:
#
# ```
# chain = {
#     "<START>": ["Some", ...],
#     "Some": ["people", ...],
#     "people": ["call", ...],
#     "call": ["me", ...],
#     "me": ["the", ...],
#     "the": ["space", ...],
#     "space": ["cowboy,", ...],
#     "cowboy,": ["yeah", ...],
#     "yeah": ["<N>", ...],
#     "<N>": ["Some", ..., "Come"],
#     ...,
#     "Come": ["on", ...],
#     "on": ["baby", ...],
#     "baby": ["and", ...],
#     "and": ["I'll", ...],
#     "I'll": ["show", ...],
#     "show": ["you", ...],
#     "you": ["a", ...],
#     "a": ["good", ...],
#     "good": ["time", ...],
#     "time": ["<END>", ...],
# }
# ```
#
# Your chain will be trained on not just one song, but by all songs by your artist.

def train_markov_chain(lyrics):
    """
    Args:
      - lyrics: a list of strings, where each string represents
                the lyrics of one song by an artist.
    
    Returns:
      A dict that maps a single word ("unigram") to a list of
      words that follow that word, representing the Markov
      chain trained on the lyrics.
    """
    chain = {"<START>": []}
    for lyric in lyrics:
        # YOUR CODE HERE
        pass
        
    return chain


# +
# Load the pickled lyrics object that you created in Lab A.
import pickle
lyrics = pickle.load(open("lyrics.pkl", "rb"))

# Call the function you wrote above.
chain = train_markov_chain(lyrics)

# What words tend to start a song (i.e., what words follow the <START> tag?)
print(chain["<START>"])

# What words tend to begin a line (i.e., what words follow the line break tag?)
print(chain["<N>"][:20])
# -

# Now, let's generate new lyrics using the Markov chain you constructed above. To do this, we'll begin at the `"<START>"` state and randomly sample a word from the list of words that follow `"<START>"`. Then, at each step, we'll randomly sample the next word from the list of words that followed each current word. We will continue this process until we sample the `"<END>"` state. This will give us the complete lyrics of a randomly generated song!
#
# You may find the `random.choice()` function helpful for this question.

# +
import random

def generate_new_lyrics(chain):
    """
    Args:
      - chain: a dict representing the Markov chain,
               such as one generated by generate_new_lyrics()
    
    Returns:
      A string representing the randomly generated song.
    """
    
    # a list for storing the generated words
    words = []
    # generate the first word
    words.append(random.choice(chain["<START>"]))
    
    # YOUR CODE HERE
    
    
    # join the words together into a string with line breaks
    lyrics = " ".join(words[:-1])
    return "\n".join(lyrics.split("<N>"))
# -

print(generate_new_lyrics(chain))


# # Question 3. Bigram Markov Chain Model
#
# Now you'll build a more complex Markov chain that uses the last _two_ words (or bigram) to predict the next word. Now your dict `chain` should map a _tuple_ of words to a list of words that appear after it.
#
# As before, you should also include tags that indicate the beginning and end of a song, as well as line breaks. That is, a tuple might contain tags like `"<START>"`, `"<END>"`, and `"<N>"`, in addition to regular words. So for example, for ["The Joker"](https://www.azlyrics.com/lyrics/stevemillerband/thejoker.html), you would add the following to your chain:
#
# ```
# chain = {
#     (None, "<START>"): ["Some", ...],
#     ("<START>", "Some"): ["people", ...],
#     ("Some", "people"): ["call", ...],
#     ("people", "call"): ["me", ...],
#     ("call", "me"): ["the", ...],
#     ("me", "the"): ["space", ...],
#     ("the", "space"): ["cowboy,", ...],
#     ("space", "cowboy,"): ["yeah", ...],
#     ("cowboy,", "yeah"): ["<N>", ...],
#     ("yeah", "<N>"): ["Some", ...],
#     ("time", "<N>"): ["Come"],
#     ...,
#     ("<N>", "Come"): ["on", ...],
#     ("Come", "on"): ["baby", ...],
#     ("on", "baby"): ["and", ...],
#     ("baby", "and"): ["I'll", ...],
#     ("and", "I'll"): ["show", ...],
#     ("I'll", "show"): ["you", ...],
#     ("show", "you"): ["a", ...],
#     ("you", "a"): ["good", ...],
#     ("a", "good"): ["time", ...],
#     ("good", "time"): ["<END>", ...],
# }
# ```

def train_markov_chain(lyrics):
    """
    Args:
      - lyrics: a list of strings, where each string represents
                the lyrics of one song by an artist.
    
    Returns:
      A dict that maps a tuple of 2 words ("bigram") to a list of
      words that follow that bigram, representing the Markov
      chain trained on the lyrics.
    """
    chain = {(None, "<START>"): []}
    for lyric in lyrics:
        # YOUR CODE HERE
        pass

    return chain


# +
# Load the pickled lyrics object that you created in Lab A.
import pickle
lyrics = pickle.load(open("lyrics.pkl", "rb"))

# Call the function you wrote above.
chain = train_markov_chain(lyrics)

# What words tend to start a song (i.e., what words follow the <START> tag?)
print(chain[(None, "<START>")])
# -

# Now, let's generate new lyrics using the Markov chain you constructed above. To do this, we'll begin at the `(None, "<START>")` state and randomly sample a word from the list of words that follow this bigram. Then, at each step, we'll randomly sample the next word from the list of words that followed the current bigram (i.e., the last two words). We will continue this process until we sample the `"<END>"` state. This will give us the complete lyrics of a randomly generated song!

# +
import random

def generate_new_lyrics(chain):
    """
    Args:
      - chain: a dict representing the Markov chain,
               such as one generated by generate_new_lyrics()
    
    Returns:
      A string representing the randomly generated song.
    """
    
    # a list for storing the generated words
    words = []
    # generate the first word
    words.append(random.choice(chain[(None, "<START>")]))
    
    # YOUR CODE HERE
    
    
    # join the words together into a string with line breaks
    lyrics = " ".join(words[:-1])
    return "\n".join(lyrics.split("<N>"))
# -

print(generate_new_lyrics(chain))

# # Analysis
#
# Compare the quality of the lyrics generated by the unigram model (in Lab B) and the bigram model (in Lab C). Which model seems to generate more reasonable lyrics? Can you explain why? What do you see as the advantages and disadvantages of each model?

# **YOUR ANSWER HERE.**

# # Submission Instructions
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
# 3. Upload the PDF [to PolyLearn](https://polylearn.calpoly.edu/AY_2018-2019/mod/assign/view.php?id=349486).
