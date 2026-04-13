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

`action-space-randomised.txt` contains the same curated action space in a randomized order. The current GA runtime
uploads the full 4,739-word catalog from this file to GPU constant memory once at process start.

`run_genetic_algorithm` then works against a configurable active prefix of that catalog. By default it starts with the
first 20 words, grows the active prefix by 1 word per generation, and keeps training-word count and selectable
action-space count equal for now. Newly activated output-embedding tails are injected on-device during next-generation
assembly.

## More Information on Data

If more information about the data is needed, see the [blog post on curating the
data](https://sam-burns.com/posts/neuroevolutionary-wordle-wordlists/).
