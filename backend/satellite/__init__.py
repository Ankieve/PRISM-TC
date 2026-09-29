"""PRISM-TC near-real-time satellite module.

See backend/satellite/README.md (and the top-level CLAUDE.md / satellite
report) for the full design, the honest compatibility findings, and how to
verify it with real internet access. This package is intentionally kept
separate from the existing, working image-upload pipeline (server.py's
/api/predict and /api/demo) so nothing that already works can be broken by
this addition. If every import in this package failed, the rest of the app
would still run exactly as before.
"""
