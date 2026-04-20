# Contributing to SCAFAD-R

## CI is the authoritative test runner

GitHub Actions (`.github/workflows/ci.yml`) runs on Ubuntu and is the
ground truth for pass/fail.  A test that is green on your laptop but red
in CI is broken.

## Git workflow

Normal development on Linux or macOS:

```bash
git add <files>
git commit -m "type(scope): message"
```

### Local NTFS escape hatch (Windows-only, rare)

When working through a Linux shell against an NTFS-mounted repository (e.g.
WSL or a remote mount), the `index.lock` / `HEAD.lock` files may become
stuck and cannot be deleted from Linux.  This is a **local-development-only
escape hatch** — it is never needed in CI:

```bash
# Stage blobs
HASH=$(git hash-object -w path/to/file)
# Build tree (see Blueprint Appendix A.2 for the full procedure)
TREE=$(git mktree < tree-spec.txt)
# Create commit
COMMIT=$(git commit-tree "$TREE" -p "$(cat .git/refs/heads/main)" -m "message")
echo "$COMMIT" > .git/refs/heads/main
```

Full procedure: **Blueprint Appendix A.2** and **Bible Playbook P-03**.
Do not treat this as standard workflow.  If you are doing this repeatedly,
switch to a Linux or macOS environment where normal `git commit` works.

## Test discipline

- Never shrink the permanent test set (T-001..T-011 + T-012).
- Every behavioural change requires a failing test committed before the
  implementation commit (Bible §8).
- `python3 scripts/audit_imports.py` must exit 0.
- `python3 scripts/class_boundary_check.py` must exit 0.
- British English in all identifiers, comments, and documentation.
