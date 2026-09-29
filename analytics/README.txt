VAYUVEGA - Data Visualisation & Analytics (Member 5)
=====================================================
What it is   : the "Analytics" section of the dashboard (index.html #analytics).
Data source  : backend/data/track_data.csv, backend/samples/samples.csv,
               inputs/reported_metrics.json (numbers quoted in CLAUDE.md).
Rebuild      : python analytics/build_analytics.py      (from the project folder, ~15 s)
Test         : python analytics/tests/test_analytics.py
Output       : analytics/analytics_data.js (+ .json) - loaded by index.html, works offline.

When Member 3 has test-set predictions
--------------------------------------
Save them as analytics/inputs/test_predictions.csv with columns
    true_class, pred_class, confidence
(class names exactly as in backend/logic.py CLASS_ORDER; confidence in percent).
Rebuild, and the Model performance card adds a confusion matrix,
per-class precision/recall/F1 and a reliability chart. Without this file those
charts are NOT drawn (no data = no chart).

Validation file
---------------
inputs/val_predictions.csv (287 images; columns filename, actual_label, predicted_label; no confidence)
is shown as a VALIDATION set. It is not the 372-image test set quoted in the notes.

Files
-----
build_analytics.py   computes every number (also scores logic.py's simulated outlook, read-only)
analytics.js/.css    charts (plain SVG) and live link to app.js via the "vayuvega:prediction" event
inputs/              reported_metrics.json, basemap.json (Natural Earth, public domain)
tools/               one-off helpers that made basemap.json
tests/               unit tests
