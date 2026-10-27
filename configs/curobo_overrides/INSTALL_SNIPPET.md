Insert these commands in installation Step 5, after extracting `curobo.zip`
and before installing CuRobo. Run them from the AffordSim repository root:

```bash
python scripts/apply_curobo_overrides.py
python scripts/apply_curobo_overrides.py --apply
cd curobo
python -m pip install -e .[isaacsim] --no-build-isolation
cd ..
```

The first command previews the changes without writing files. `--apply`
installs the versioned stock Franka FR3 TCP and joint-limit corrections into
this checkout's `curobo/` directory. Differing existing files are backed up
under `outputs/curobo_override_backups/`; files already matching the overrides
are left untouched. Start a fresh collection process after applying changes.

The override files retain their paths relative to `curobo/`. Their SHA-256
checksums and the corresponding bundled-archive checksums are recorded in
`manifest.json`. Keep the manifest checksums synchronized if an override is
intentionally edited. The complete CuRobo source remains supplied by
`curobo.zip`; this overlay contains only the two corrected robot files.
