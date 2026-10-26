<!--
Thank you for your contribution to samurai!

How to fill in this pull request (people and AI agents alike):

1. Run `pre-commit run --all-files` and commit what it fixes.
2. Write the title in the conventional commits form
   (https://www.conventionalcommits.org/en/v1.0.0/): `type(scope): summary`,
   imperative, lower case, no final period. It becomes the commit message on `main`.
   Types: build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test.
   Add `!` after the scope when the PR breaks the public API: `feat(io)!: ...`.
3. Fill in the sections below for a reviewer who knows samurai but not this change.
   Keep the whole text under about 250 words.
4. Delete every section that has nothing to say (never write "N/A"),
   then delete all these comments.

For each sentence, ask: does it change what the reviewer checks, or what a user
has to do? If not, cut it.

- Name functions, files, demos, options and error messages exactly, in backticks.
- Give results as numbers, a small table or a before/after, not adjectives.
- Leave out what the diff, the CI or the title already shows: the list of changed
  files, index formulas and loop bounds, renamed private helpers, "all tests pass".
- Leave out the story of the work (what you tried first) and any mention of the
  tool or agent that wrote the change. An alternative a reviewer would suggest
  is worth one line: "no cache here: `X` invalidates the pointers".
-->

## Description

<!--
What was wrong or missing, and what the PR changes.
- fix: the symptom as a user hits it (command, error message, wrong value), the cause, the fix.
- feat: what a user can now do, with a short code example.
- refactor/perf: what changes in the code structure or the timings; say whether results change.
Put the evidence for this change next to the claim it supports: a reproduction that
confirms the cause, outputs identical to `main` after a refactor, a timing.
-->

## Breaking changes

<!--
What user code stops compiling or behaves differently, and how to migrate
(a before/after snippet). Examples: a public function removed, renamed or moved
to `samurai::detail`, a type no longer copyable, a changed default or signature.
-->

## Related issue

<!-- `Closes #123`, or where the problem was found (issue, PR, CI job). -->

## How has this been tested?

<!--
Only the tests this PR adds or changes: their name, what they check, and whether
they fail without this PR. The CI runs the existing suite: do not report its results.
-->

## Reviewer notes

<!--
A debatable design choice, a risky area, where to start reading a large PR,
or a code path nobody ran (MPI, a platform) that the PR touches.
-->
