# AGENTS.md — Adaptive Product Matching

## Mission

This repository supports a master's thesis titled **"Інтелектуальна система адаптивного пошуку товарних пропозицій"**. Build a maintainable, reproducible local product-offer search application while preserving research integrity. Act as a careful senior Python engineer and skeptical code reviewer. Challenge proposals that add complexity without measurable value.

**Read first:** `docs/technical_specification.md`, `README.md`, relevant files in `docs/final_evaluation/` and `docs/search_system/` (if present), and the existing tests. Treat the technical specification as the target state; verify existing code rather than assuming the target is implemented. If a referenced file does not exist locally, say so instead of inventing its content.

## Context and boundaries

- Target environment: Windows, Python 3.12.10, existing `.venv` where available.
- Current repository includes experimental matching/evaluation scripts and a query-by-example local search prototype. Verify exact modules and behavior before refactoring.
- Approximate entrypoints to inspect: `product_catalog.py`, `search_system.py`, `search_cli.py`, `matching.py`, `adaptive_corrected.py`, `logistic_baseline.py`, `fpv_dataset.py` and `tests/`.
- Methods studied: TF-IDF, Fixed Weighted, Renormalization, Adaptive v1, Logistic Regression, Adaptive corrected. Do **not** characterize Adaptive v1 as the proven best or novel algorithm.
- Existing WDC TEST pairwise results have already been inspected. **Never tune using that TEST and then claim independent confirmation.** Pairwise Precision/Recall/F1 do not establish candidate retrieval or end-to-end ranking quality.
- FPV data is a small auxiliary set. Its few proposed pair labels are not an independently verified quantitative benchmark.
- Product technical fields may be stored as metadata without contributing to scoring. Never quietly promote metadata into scoring features.

## Working protocol (mandatory)

1. **Inspect before modifying.** Examine current files, imports, tests, docs, branch/working tree, generated artifacts and relevant call sites. Locate existing functionality and avoid duplicating it.
2. **Establish a baseline.** Report the commands and results of tests/lint/type checks available in the existing project. If tests fail or dependencies/data are missing, record the limitation. Do not claim they passed.
3. **Plan in small stages.** For substantial work, describe scope, affected modules, behavior to preserve, tests, risks and a rollback path. Get user confirmation before broad structural moves or changes to research behavior.
4. **Add regression protection first.** Write characterization tests for observable current behavior when necessary, then refactor in reviewable steps. Do not use a single giant rewrite.
5. **Keep research code/data separate from presentation.** The search service must be shared by CLI and web UI; the UI must not reimplement matching/scoring. Gold labels must not flow into the search engine.
6. **Verify each stage.** Run focused tests, then the full available suite. Report exact commands, pass/fail counts, scope and unverified cases. Compare numerical outputs with baseline when relevant.
7. **Summarize facts, not impressions.** State what changed, why, evidence it still works, known limitations and next step. Distinguish existing/implemented/tested/planned/hypothetical features.

## Research integrity — hard constraints

- Do not invent experiments, results, statistical significance, novelty, features or evaluation data.
- Preserve historical scripts, frozen artifacts, configurations, reported scores, train/validation/test splits and documentation provenance. Do not rewrite research history to make outputs prettier.
- Do not fit, train, retune, select thresholds or calibrate on the reviewed WDC TEST. If a new evaluation uses TEST, clearly label it as post-hoc/exploratory and describe limitations; prefer development datasets for design decisions.
- Audit qrels before information-retrieval evaluation: verify offer IDs, cluster IDs, label consistency, completeness, leakage and singleton queries. No assumption that a pairwise file is a full catalog-level ground truth.
- Evaluate retrieval and ranking separately. For a fair comparison, rerankers receive the same candidate pools, queries and evaluation protocol.
- In missing-attribute experiments, apply masking at offer level before retrieval and reranking, use fixed seeds and identical scenarios across methods.
- Missing field means missing, not zero or a related/inferred attribute. `score` is not necessarily an identity probability.
- Do not replace proven behavior with arbitrary model-code penalties, numeric-attribute penalties or Fixed/Renormalization switching rules. Revisit only with a new explicit rationale and an approved protocol.

## Code quality and architecture

- Prefer Python 3.12 typing, cohesive functions, explicit dependencies, sensible exceptions and small focused modules. Use existing conventions unless there is a strong reason to change them.
- Avoid unnecessary classes, abstractions, new packages, dependencies and speculative scalability engineering.
- Keep a stable compatibility surface for current CLI, JSON output, tests and experiment scripts unless the user approves a documented breaking change.
- Refactor without changing normalization, fitted TF-IDF conventions, score formulas, default ranking/tie-break ordering or frozen model provenance. Any such change is a separate **research change**, not a refactor.
- Reuse `ProductOffer`, the candidate retriever, existing feature extraction and scoring. Do not create parallel copies in the UI.
- Treat dataset paths as configurable; do not embed machine-specific absolute paths or assume untracked WDC/frozen files exist.
- Test deterministic edge cases: empty catalog/query, missing titles and sources, duplicates, unavailable model artifacts, zero candidates, stable ordering, unknown ID, validation errors, incomplete attributes.
- Prefer robust minimal solutions over adding a database, background workers, authentication, vector store or SPA when not required.

## Web UI

- A polished **local** web interface is in scope. It must support selecting a real catalog offer (by title lookup/ID), running query-by-example search, choosing available methods/limits, displaying ranked cards and comparing real product fields.
- UI selection by text is not automatically a free-form semantic search feature. Do not claim live internet search.
- Recommended starting option after inspection: FastAPI + Jinja2 + lightweight CSS/JS. An alternative such as Streamlit is allowed only with a short written architectural rationale and user approval. Do not build a second scoring pipeline.
- Hide unavailable methods unless a compatible verified frozen artifact exists. Handle missing fields and unknown sources honestly. Show score explanations only where the scorer exposes them.
- Keep UI accessible, responsive and testable; document a single local launch command. Do not invent product photos, prices, URLs or external-source metadata.

## Git ownership, commits, and mandatory GitHub main synchronization

The user has **explicitly delegated routine Git history management and authorized pushing completed, approved changes to the remote default branch `origin/main`**. This is a standing instruction: **do not ask for separate permission for each ordinary push** once the task/implementation stage has been approved and its acceptance criteria are met. The goal is for all approved, verified, publishable project changes to reach the GitHub `main` branch, not to remain indefinitely on local feature branches.

- **Start safely:** inspect `git status`, current branch, remotes, recent commits, staged diff and `origin/main` before making changes. Do not overwrite, discard, auto-stage or commit unrelated user edits. If ownership or intent of existing changes is uncertain, ask before including them.
- **Work in small, reviewable steps:** a temporary feature/refactor branch is allowed and encouraged for risky work. Commit completed, verified steps with descriptive messages (`test:`, `refactor:`, `feat:`, `docs:`, `fix:`). Inspect staged files and diffs before every commit; stage only intended, permissible files.
- **Publish to `main` by default:** after the user has approved the stage, acceptance criteria are met and relevant checks pass, integrate its commits into local `main` using normal non-destructive Git operations, then push to **`origin main`**. Do not consider a task fully delivered while its approved commits exist only on a topic branch. Do not push the topic branch instead of `main` unless explicitly requested or required by repository protections.
- **Remote safety:** fetch/check the remote state before integration and pushing. Preserve any commits already present on `origin/main`. If local/remote history diverges, a merge conflict occurs, checks fail, the remote is inaccessible, or GitHub branch protection requires a pull request, **stop and explain the blocker**; do not bypass protections, force-push, silently rebase, or claim a successful push. When a pull request is required, use that workflow and clearly state that `main` is not yet updated.
- **Verify the push:** confirm that `origin/main` points to the expected published commit(s), then report the branch, commit SHA(s), test results and whether the push succeeded. An unverified command invocation is not proof of publication.
- **Protect research reproducibility:** frozen hashes, recorded manifests and test snapshots are historical evidence. Preserve file-specific line endings, especially where the audit found LF/CRLF-sensitive hashes. Do not change global Git normalization, bulk-normalize files or rewrite historical manifests to make checks pass without a separately approved and validated migration.
- **Never commit or publish prohibited material:** secrets, `.venv`, personal settings, raw copyrighted/large datasets, untracked frozen artifacts, generated caches or other excluded files. Review `.gitignore`, staged contents and diffs before every commit/push. "Push all changes" means all **approved, verified and permissible source/documentation changes**, not every untracked file on disk.
- **Forbidden without explicit additional permission:** `git push --force` (including `--force-with-lease`), history rewriting, destructive resets/cleans, deleting branches, amending others' commits or bypassing safeguards. Do not clone when already in the supplied checkout.

## First milestone: baseline + controlled refactoring

This milestone comes **before new product features**. Audit the complete project, map dependencies and document a prioritized refactoring plan. Protect current search behavior with tests; execute incremental refactors only after the plan is approved. Maintain reproducibility and keep experimental scripts runnable. Do not implement the web UI or tune models as part of the first refactor milestone.

## Definition of done for a change

- The intended change and acceptance criteria are satisfied.
- New/changed behavior is covered by appropriate tests; known existing failures are clearly reported.
- The existing CLI/search and experimental reproducibility are not silently broken.
- Documentation and usage commands are updated when needed.
- The final summary reports files changed, tests executed, outcomes, limitations, commit ID(s) and GitHub `origin/main` publication status.
- Approved, verified changes are integrated into `main` and pushed to `origin/main`, or a concrete blocker is explicitly reported.
- No prohibited data or artifacts were added to Git.
