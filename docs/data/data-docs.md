# Data in the Project

## Files

Regarding the data files in `data/`, these are:

### Allowed Guesses

`allowed-guesses.txt` is all of the legal guesses in a Wordle. There are 14,855 of these. This file is essentially not
being used in the project, and is included only for completeness.

### Allowed Solutions

`allowed-solutions.txt` are the 2,309 allowed solutions to a Wordle puzzle. They are often not pluralised nouns or 3rd
person verbs with an _-s_ inflection. They are usually words that any English speaker would know.

### Most Common Subtitles

`most-common-subtitles.txt` are the 5,000 most commonly used 5-letter words in subtitles on American television.

### Action Space

`action-space.txt` are 4,739 words that the Wordle player policy may output. This is a stripped down version of the
allowed guesses, made smaller to make training faster. All of these words are in the allowed guesses. Every allowed
solution is included in this file.

`action-space-randomised.txt` contains the same curated action space in a randomized order. The current GA runtime uses
the top 20 words from this file as its training shard and uploads that shard to GPU constant memory while the run is in
progress.

## More Information on Data

If more information about the data is needed, see the [blog post on curating the
data](https://sam-burns.com/posts/neuroevolutionary-wordle-wordlists/).
