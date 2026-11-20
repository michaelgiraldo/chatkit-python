# Turnspan upstream status

## Turnspan 2 audit

This November 19, 2026 snapshot covers all open issues and pull requests in
OpenAI's ChatKit Python repository. We checked them against Turnspan 2
(`1.6.5+turnspan.2`, commit `4c7ba49`) on Python 3.14. The release passed 164
tests, and its GitHub Actions run passed. The table also counts fixes verified
by CI on Turnspan's maintained Python 3.14 branch after that release.
[Open issues](https://github.com/openai/chatkit-python/issues) ·
[Open pull requests](https://github.com/openai/chatkit-python/pulls) ·
[Turnspan 2 release](https://github.com/michaelgiraldo/chatkit-python/releases/tag/v1.6.5%2Bturnspan.2) ·
[Release CI](https://github.com/michaelgiraldo/chatkit-python/actions/runs/44643617666)

| Decision | Issues | PRs | Entries | Distinct topics |
| --- | ---: | ---: | ---: | ---: |
| Resolved on Turnspan | 5 | 13 | 18 | 13 |
| Deferred | 2 | 1 | 3 | 3 |
| Reviewed, no fix needed | 4 | 1 | 5 | 5 |
| **Total reviewed** | **11** | **15** | **26** | **21** |

The maintained branch resolves **18 of 26 entries (69%)** across **13 of 21
topics (62%)**. Turnspan 2 shipped fixes for 15 entries across 11 topics. The
test collection and web-search progress fixes landed after the release.
Entries count individual GitHub reports and patches. Topics count each
underlying behavior once. The fixed count includes code, docs, and tooling
changes that fully address the reported need.

## Resolved in Turnspan 2

| Topic | What we verified and shipped |
| --- | --- |
| Stream cleanup after disconnect | In the early-close reproduction, upstream left a pending stream read and Turnspan cleaned it up. The HTTP regression also verifies that the partial reply reloads and the thread accepts a new turn. [Issue #138](https://github.com/openai/chatkit-python/issues/138) · [PR #133](https://github.com/openai/chatkit-python/pull/133) · [Cleanup](https://github.com/michaelgiraldo/chatkit-python/commit/f2c64cc) · [HTTP regression](https://github.com/michaelgiraldo/chatkit-python/commit/c1494e3) |
| Distinct Chat Completions reply IDs | Two replies keep separate IDs from streaming through thread reload. The Responses path preserves the IDs supplied by OpenAI. [Issue #36](https://github.com/openai/chatkit-python/issues/36) · [PR #128](https://github.com/openai/chatkit-python/pull/128) · [Fix and tests](https://github.com/michaelgiraldo/chatkit-python/commit/8e18276) |
| ListView inside a Card | Both PRs address the same missing widget type. The release preserves typed rows and click actions through serialization. We also rendered the nested list, clicked a row, and reloaded the thread in a browser. [Issue #47](https://github.com/openai/chatkit-python/issues/47) · [PR #46](https://github.com/openai/chatkit-python/pull/46) · [PR #137](https://github.com/openai/chatkit-python/pull/137) · [Fix and test](https://github.com/michaelgiraldo/chatkit-python/commit/c581f98) |
| Streamed citations | The converter accepts the SDK's dictionary-shaped citation payloads. A regression verifies that an empty citation is skipped and the next valid citation still arrives. [PR #129](https://github.com/openai/chatkit-python/pull/129) · [Citation conversion](https://github.com/michaelgiraldo/chatkit-python/commit/c650ce6) · [Empty citation regression](https://github.com/michaelgiraldo/chatkit-python/commit/785cca6) |
| Text updates without component IDs | Tests confirm that text appended to static and dynamic widgets without IDs reaches the client through a replacement update. [PR #132](https://github.com/openai/chatkit-python/pull/132) · [Fix and tests](https://github.com/michaelgiraldo/chatkit-python/commit/ee764e4) |
| Added or removed widget properties | Tests confirm that both added and removed dynamic properties produce a replacement with the correct widget data. [PR #134](https://github.com/openai/chatkit-python/pull/134) · [Fix and tests](https://github.com/michaelgiraldo/chatkit-python/commit/362b1e3) |
| Equal workflow tasks get distinct positions | Appending a task uses its new position even when an earlier task has the same title and content. Tests cover equal and different tasks. [PR #135](https://github.com/openai/chatkit-python/pull/135) · [Fix and tests](https://github.com/michaelgiraldo/chatkit-python/commit/b328f9b) |
| Agents SDK header overrides | Removing automatic User-Agent tagging eliminated Turnspan's temporary override. Tests checked the SDK default header on Responses and an app-configured header on Chat Completions. [PR #136](https://github.com/openai/chatkit-python/pull/136) · [Removal and tests](https://github.com/michaelgiraldo/chatkit-python/commit/58f57af) |
| Quickstart backend verification | The quickstart now includes a curl smoke test. We ran its request against the documented FastAPI backend without an API key and got a completed event containing `Hello, world!`. [PR #118](https://github.com/openai/chatkit-python/pull/118) · [Quickstart smoke test](quickstart.md#verify-your-backend) · [Docs change](https://github.com/michaelgiraldo/chatkit-python/commit/1c0b0a2) |
| Structured output persistence | The guide shows how to save structured tool results after a successful stream, with an upsert keyed to the thread and user message. A scripted Agents SDK run exercised a successful save and verified that a failed turn skips it. [Issue #120](https://github.com/openai/chatkit-python/issues/120) · [Persistence example](guides/respond-to-user-message.md#save-structured-tool-results-after-a-turn) |
| GitHub Actions setup-uv update | The PR proposes setup-uv v7. Turnspan pins the newer v10.3.3 by commit in its CI, docs, and release workflows. The exact release commit passed CI. [PR #84](https://github.com/openai/chatkit-python/pull/84) · [Workflow pins](https://github.com/michaelgiraldo/chatkit-python/commit/59aff6e) |

## Resolved after Turnspan 2

| Topic | What we verified on the Python 3.14 branch |
| --- | --- |
| Test imports in a clean checkout | **Fixed.** Shared server and stream helpers live in `tests/helpers/`, and plain `pytest` defaults to `importlib` mode. All nine test modules collected individually, and CI passed all 164 tests. Turnspan 2 had already removed the private Agents SDK sentinel import. Thanks to Mark Zhang for reporting the import problem. [PR #122](https://github.com/openai/chatkit-python/pull/122) · [Test setup](https://github.com/michaelgiraldo/chatkit-python/commit/92a1696934836e72469d94e858ee322b58f0415c) · [Passing CI run](https://github.com/michaelgiraldo/chatkit-python/actions/runs/44663927249) · [Fixture cleanup](https://github.com/michaelgiraldo/chatkit-python/commit/58f57af) |
| Web-search progress | **Fixed.** Each Responses search keeps its own task and completion state by provider item ID. The regressions exercise overlapping searches, cancellation, and final citations. Chrome replay with synthetic Responses events verified independent progress and completion, a working Stop button, and the final citation inline and in the Sources drawer. Thanks to Francis Poisson-Gagnon for the report and proposed patch. [Issue #28](https://github.com/openai/chatkit-python/issues/28) · [PR #29](https://github.com/openai/chatkit-python/pull/29) |

## Deferred

| Topic | What we verified and why it is deferred |
| --- | --- |
| Mypy checks on the SDK source | Mypy 2.4.1 on Python 3.14 reports 27 errors in three Turnspan source files. A smoke check importing public types from an installed Turnspan wheel passes, as does the supported Pyright check. Revisit if an app demonstrates a typing blocker or Turnspan adopts mypy as a supported checker. [PR #7](https://github.com/openai/chatkit-python/pull/7) |
| Large inline PDF persistence | The mocked transport captured an inline PDF in a request to `/v1/responses` and zero Files API requests. The hosted service's storage behavior and the reported size threshold still need verification on the Responses API side. [Issue #109](https://github.com/openai/chatkit-python/issues/109) |
| Report without a ChatKit reproduction | The report provides no ChatKit behavior or reproduction steps to test. Revisit when an example identifies a failing SDK operation. [Issue #130](https://github.com/openai/chatkit-python/issues/130) |

## Reviewed, no fix needed

| Topic | What we verified and why Turnspan needs no change |
| --- | --- |
| Node.js and TypeScript backend | The issue requests a separate server SDK. The maintainer's reply discusses an official Node backend and community ports. Turnspan maintains the Python server SDK. [Issue #5 and maintainer reply](https://github.com/openai/chatkit-python/issues/5#issuecomment-3514151795) |
| Custom browser components | The maintainer's reply identifies browser rendering as the area that would need support for custom UI. Any Python-side schema change would need a supported renderer contract first. [Issue #53 and maintainer reply](https://github.com/openai/chatkit-python/issues/53#issuecomment-3514071827) |
| Citation formatting in hosted Agent Builder | The reporter confirmed that the problem occurs in hosted Agent Builder. Turnspan's self-hosted citation fix addresses the SDK stream described in PR #129. [Issue #124 and reporter clarification](https://github.com/openai/chatkit-python/issues/124#issuecomment-4330017494) |
| OpenAI maintenance status and roadmap | This is a question for the maintainers of the original repo. Turnspan can publish its own release and triage status. The issue describes no SDK behavior to change. [Issue #139](https://github.com/openai/chatkit-python/issues/139) |

### PR #21 — API reference URLs and Pylint

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
