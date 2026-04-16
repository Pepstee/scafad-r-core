Archive layout for local repository preparation.

Purpose:
- keep the active repo root focused on current Layer 0 and Layer 1 work
- retain older/generated/runtime material locally for reference
- make it obvious which files Claude can ignore on resume

Folders:
- `generated/`: build output, telemetry captures, starter-template material, scratch workspace
- `reports/`: historical summaries, one-off reports, and root-level analysis writeups
- `runtime_artifacts/`: logs, CSVs, assessment outputs, validation result files
- `root_test_scripts/`: older root-level test scripts that are no longer the canonical test entrypoints
- `helper_scripts/`: ad hoc execution and analysis helpers moved out of the main root
- `fixtures/`: archived root-level event payloads
- `notes/`: collaborator-specific notes that are not part of the main code path

Active test entrypoints now expected at repo root:
- `tests/test_l0_l1_contract.py`
- `tests/test_001_l0_l1_smoke.py`
