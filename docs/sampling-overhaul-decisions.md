# Sampling Overhaul — Findings & Rationale (2026-09-26)

**Summary:** Three sampling fixes shipped (thread-local candidate scratch, per-thread mt19937, params fingerprint); pooled-chain deferred behind a J7b measurement gate.

## Why (measured, vs LLamaSharp baseline ~0.5ms/token)
1. **Candidate array, not chain init, was the dominant cost.** Hot path rebuilt a 150k-entry `std::vector<llama_token_data>` per draw attempt (up to 16 with ignore_eos) plus a full-vocab grammar mask per attempt. → Fixed with `t_candidates()` thread-local scratch: buffer sized once to n_vocab, ids written once, only logits rewritten per attempt.
2. **`rand()` was global + thread-unsafe.** Batched server samples on multiple threads; `srand` shares one seed state and `rand()` is a data race with poor distribution. → Per-thread `std::mt19937` (`t_rng`) seeded once from `random_device`; seeds dist sampler via `(uint32_t)t_rng()`.
3. **`params_fingerprint()`** (FNV-1a over temp/top_k/top_p/min_p/ignore_eos): groundwork for the future pooled selection chain — any param change forces rebuild. Deliberately NOT fingerprinted: dist seed (per-call by design), penalties (history-feeding → rebuild anyway).

## Why NOT pool the selection chain (verdict)
- Selection-chain params are per-request; `recent_tokens` changes every token; rewind/shift/lease-return force rebuilds. Persisting buys single-digit µs, costs bookkeeping.
- Grammar chain (Tier-1, conversation-owned) stays persisted — that's a parse-state machine, already correct in `ensure_grammar_chain`.
- Architecture now mirrors llama-server: grammar masks FIRST on full vocab, selection chain per call, dist last.

## Gated future step
If J7b micro-measurement shows chain init ≥ ~5% of sampling time → pooled chain with exclusions (penalties, dist, grammar), guarded by `params_fingerprint`. Until measured: do NOT implement.

## Invariants (do not break)
- Grammar mask applied FIRST, from the persistent conversation-owned chain
- ignore_eos re-draws start from unmodified raw logits snapshot (never feed EOG into grammar accept)
- EOG/control tokens never fed to `llama_sampler_accept`

**Status:** implemented in `src/ecainference.cpp` (see t_rng/t_candidates/params_fingerprint); commit pending.
**Added:** 2026-09-26
