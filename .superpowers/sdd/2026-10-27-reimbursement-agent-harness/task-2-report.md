# Task 2 report — governed reimbursement tools

## Scope and files

- Added `tools/reimbursement_tool.py`: nine deterministic demo tools, exposed only to the billing Agent through `MCPToolManager.call`.
- Updated `tools/loader.py`: optional repository dependency injection; existing no-argument tool factories and discovery behavior remain available.
- Updated `api/main.py`: lifespan creates one process-local repository and injects it into tool discovery.
- Updated `core/reimbursement_repository.py`: one public, repository-locked `submit_claim` method that validates current case state and amount, checks ownership and budget, reserves budget, assigns one claim number, persists state and idempotent result atomically.
- Added `tests/test_reimbursement_tools.py`: manager-level tool tests; handlers are never called directly.

## RED / GREEN evidence

- RED: `.venv\Scripts\python.exe -m unittest tests.test_reimbursement_tools -v` exited 1 with four `TypeError: discover_agent_tools() got an unexpected keyword argument 'dependencies'` errors before implementation.
- RED for security/workflow edge cases: `.venv\Scripts\python.exe -m unittest tests.test_reimbursement_tools.ReimbursementToolsTests.test_idempotent_result_cannot_be_read_by_another_user tests.test_reimbursement_tools.ReimbursementToolsTests.test_payment_requires_finance_review -v` exited 1 with two assertion failures (cross-user idempotent result returned; payment without review succeeded).
- GREEN: `.venv\Scripts\python.exe -m unittest tests.test_reimbursement_tools tests.test_agent_tools -v` exited 0: 18 tests run, 1 skipped (`API dependencies not installed`). Expected rejection tests produce tool-error log lines.
- `git diff --check` exited 0. Git emitted an LF to CRLF warning for the repository file, with no whitespace error.

The first sandboxed asyncio unittest command hung after printing the first test name. I interrupted it and reran the exact test command with focused elevated execution. All later asyncio test runs used that same focused exception. No packages or files were installed outside the virtual environment.

## Self-review

- `submit_claim` requires trusted `confirmed is True`, checks trusted owner and conversation before looking up a persisted result, and is uncacheable. Distinct concurrent keys cannot reserve twice. Invalid amounts, hotel over-limit, missing facts, absent expenses, missing schema parameters, unauthorized Agent, insufficient budget, and missing finance review have tests.
- Read-only `search_policy`, `get_case`, and `check_budget` are cacheable as required; write tools have zero cache TTL.
- Review approval is persisted in the case validation data; simulated payment requires it and a finance role.
- No ERP, OCR, external integration, or new third-party dependency was added.

## Concerns

- Existing `MCPToolManager` caches read results by tool name and params only, without caller context. Thus cached `get_case` and `check_budget` responses could be reused across different callers who supply the same case ID during the short TTL. Fixing this safely requires a manager-level context-aware cache key or bypass; that file is outside this task's authorized edit scope. The controller should address this before exposing case reads to untrusted callers.
- The API serialization test in the existing suite was skipped because optional API dependencies are not installed in this worktree's virtual environment.

## Review fixes (follow-up commit)

The three Important findings were reproduced before changes:

1. `.venv\Scripts\python.exe -m unittest tests.test_reimbursement_tools -v` exited 1 (12 tests, 7 failures). The cache tests showed the same params returning a cached result to another user or conversation. `MCPToolManager._cache_key()` used only tool name and params, before handler authorization.
2. The same RED run showed a rejected claim left the engineering budget at 2200.0 instead of restoring 3000.0. The old rejection path changed only case state with `save_case()`.
3. The same RED run showed NaN/infinity accepted by validation or submission, including a NaN budget after direct repository submission.

Fixes:

- `mcp/tool_manager.py` now incorporates trusted `user_id` and `conv_id` in cache keys whenever context is supplied; calls without context retain the previous key behavior. Manager-level tests cover a cache hit for the same identity, misses for another user and conversation, and a hit with absent context. `get_case` and `check_budget` tests also verify denied cross-identity calls after a cache entry exists.
- `core/reimbursement_repository.py` records the exact reserved amount per case and exposes `reject_claim()`, which releases it and changes state under the same lock, once. `review_claim` routes rejection through this method. The regression test submits, rejects, confirms duplicate rejection fails without a second refund, revalidates, and resubmits with a new key; final budget is 2200.0.
- Tool validation rejects nonfinite individual amounts and overflowed totals. Repository submission independently rejects nonfinite or nonpositive required budget and nonfinite available budget before any mutation. Tests cover NaN, positive/negative infinity, a finite unchanged budget, and the repository invariant.

GREEN evidence: `.venv\Scripts\python.exe -m unittest tests.test_reimbursement_tools tests.test_agent_tools tests.test_reimbursement_models_repository -v` exited 0: 30 tests run, 1 skipped (`API dependencies not installed`). `git diff --check` exited 0. The focused test command ran elevated because the initial sandboxed asyncio unittest had hung earlier in this task. No dependencies were installed.

Self-review: cache-key changes affect only calls with context; a rejected reservation cannot be released twice because the method requires `SUBMITTED` plus `budget_reserved`, and records the original amount rather than recomputing it from editable claim data. Submission amount is checked before budget mutation. The cache-leak concern in the original report is resolved by this follow-up. The optional API serialization test remains skipped in this environment.
