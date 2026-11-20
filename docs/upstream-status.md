# Turnspan upstream status

## Turnspan 2 audit

We reviewed all 26 open upstream entries in this audit snapshot (11 issues and
15 pull requests). Turnspan 2 addresses 16 entries, or 12 of 21 distinct topics
after grouping reports and pull requests about the same behavior. The count
includes PR #122 because plain `pytest` works. The optional `importlib` mode
still has collection errors. Counting that gap separately gives 15 fully
addressed entries and one partial.
[PR #122](https://github.com/openai/chatkit-python/pull/122) ·
[Open issues](https://github.com/openai/chatkit-python/issues) ·
[Open pull requests](https://github.com/openai/chatkit-python/pulls) ·
[Turnspan 2 release](https://github.com/michaelgiraldo/chatkit-python/releases/tag/v1.6.5%2Bturnspan.2)

## PR #21 — API reference URLs and Pylint

**Reviewed, no fix needed.** We kept `/api/chatkit/` because the MkDocs navigation,
the reference overview, and the guides link to modules at that path. It also
matches OpenAI’s published reference. The PR proposes moving the generated
pages to `/chatkit/chatkit/`. Its separate Pylint workflow targets Python
3.8–3.10, below Turnspan’s Python 3.14 baseline.
[PR #21 diff](https://github.com/openai/chatkit-python/pull/21/files) ·
[Published API reference](https://openai.github.io/chatkit-python/api/chatkit/)

We ran `mkdocs build --strict` in two clean copies of Turnspan 2 (`4c7ba49`).
The release passed with zero build warnings. Applying only the proposed URL
change moved generated pages such as `server/` to the new path. The sidebar and
guide links still pointed to the old paths. The strict build failed with
45 warnings. PR #21 is excluded from the fixed count.
