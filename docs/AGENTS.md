# Instructions to Coding Agents for the Neuroevolutionary Wordle Project

## References

If you haven't read the main `README.md` file, please do so. For coding relating to the neural net component of the
project, please see `docs/neural-net-design.md` for more information.

## Tool Overview

All there is to be said about this initially is that we are using CUDA. Target Nvidia Compute Compatibility level is
12.0. Target hardware is an RTX 5070 Ti GPU with 16Gb of VRAM.

## Coding Guidelines

Writing unit tests first with Test Driven Development may help the agent, or it may not. A coding agent should think
about how much this suggestion should affect its approach, which could be not at all.

Well commented code is good, and files should be self-documenting where possible. Adding additional markdown files with
more documentation is acceptable sometimes, but a coding agent may choose to check with the user.

Commit messages can be in any structure, and may be a large paragraph or a number of bullet points. If a change is small
and the commit message can be small, then this structure is preferred:

`To/Because/For [reason for change], [nature of change].`

E.g.:

` To begin specifying the project design, add design docs`

When implementing a feature, a coding agent should not go too far beyond the changes that were asked for. The agent will
not be asked to one-shot the entire project, so take things one step at a time.

## Interacting with the User

The user can be spoken to quite directly and bluntly, if a proposed change is a bad idea. In general, the user's
knowledge of genetic algorithms is very good. Understanding of neural nets is a little weaker, but not that bad. CUDA
experience is much weaker, and the user will benefit from guidance. The user wants idiomatic CUDA, with a sane project
structure, and may from time to time benefit from guidance on how to achieve this. Understanding of the architecture and
memory structure of a GPU is present, but with opportunities for improvement, so be clear if part of the project needs
to change.

Conversationally, the user does not like receiving very long messages unless you are told otherwise. The user prefers to
take things step-by-step, with opportunities to ask questions as we go. The user will often be clear if they are asking
for a high-level overview or a more detailed answer, and the coding agent should respect their wishes.

## Systems Administration

Coding agents are not to act as sysadmins, with regards to the devlopment environments in which they run. It is not the
agent's job to fix it if, for example:
- a compiler is missing
- the $PATH is wrong
- a dependency management tool which is needed cannot be found.

Agents MUST NOT attempt to fix such problems. Your job is not to install things. Please escalate any such issues to the
user, who will do the necessary systems administration for you.
