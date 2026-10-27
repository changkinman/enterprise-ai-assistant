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
