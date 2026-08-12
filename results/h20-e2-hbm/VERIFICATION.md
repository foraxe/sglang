# Verification

- Fixed-seed full model ABBA x 5: 20/20 rows completed; output hash identical.
- DP-rank-stratified HBM: exact native/coTensor parity, `0 MiB` delta.
- Synthetic E1 rerun: native and coTensor `PASS`, zero mismatches, identical
  four full-byte SHA-256 values, 200 prefetch operations per rank.
- Local targeted suite: `15 passed` covering 11 setup/cleanup transaction
  cases, late POSIX-FD timeout ownership, and backend selection.
- Existing TTFT/prefill evidence: `-0.399%` and `+0.172%`, both within 3%.
- Final postflight: all eight GPUs `1 MiB`, `0%`; no experiment processes.
