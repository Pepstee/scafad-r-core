---
status: PENDING
last_updated: 2026-04-26
recording_location: null
file_bytes: null
sha256: null
duration_seconds: null
recorded_by: null
---

# Showreel Recording Status

**Current status:** PENDING — the screen-recording walkthrough has not yet been committed to the repository.

---

## What this file tracks

This file is the single source of truth for the lifecycle of the SCAFAD demonstration recording.
It transitions through three states:

| State | Meaning |
|-------|---------|
| `PENDING` | Recording not yet produced or committed |
| `RECORDED_LOCAL` | MP4 committed to `presentation/showreel.mp4` (≤ 50 MB) |
| `RECORDED_EXTERNAL` | Recording hosted externally; URL in `presentation/showreel_url.txt` |

---

## Next steps (for the human operator)

The narrated script, ordered shot list, and full capture checklist are in
`presentation/showreel.md`.  Follow those instructions to record the
demonstration, then either:

- **≤ 50 MB recording:** commit `presentation/showreel.mp4` directly and update
  this file's front-matter (`status: RECORDED_LOCAL`, populate `file_bytes`,
  `sha256`, `duration_seconds`, `recorded_by`).

- **> 50 MB recording:** place the HTTPS URL in `presentation/showreel_url.txt`
  (see `presentation/showreel_url.txt.template` for the required format) and
  update this file's front-matter (`status: RECORDED_EXTERNAL`, populate
  `recording_location` with the URL).

---

## Verification

Once the recording is committed, run:

```bash
python scripts/verify_showreel.py
```

Expected output:

- `LOCAL  <bytes>  <sha256>` — recording is local and valid.
- `EXTERNAL  <url>` — recording is hosted externally.
- `PENDING …` — recording not yet present (current state).

---

*Maintained by the SCAFAD build pipeline · last updated 2026-04-26*
