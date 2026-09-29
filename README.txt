PRISM-TC  -  Frontend + Backend (SIH26070)
==========================================

Folder structure
----------------
PRISM-TC-Frontend/
  index.html, style.css, app.js   the dashboard (Member 4's design, extended)
  backend/
    server.py        web server: serves the dashboard AND the /api routes
    logic.py         track / risk / class logic
    predict.py       the AI model call (needs model.pth, NO normalization)
    model.pth        <-- YOU MUST COPY THIS IN (final model from Drive scripts/)
    data/            historical storm positions (from IBTrACS)
    samples/         10 test images for the demo
    tests/smoke_test.py
    requirements.txt
  CLAUDE.md          instructions/notes for Claude (read this first if you use Claude)
  _original_frontend_backup/   Member 4's original files, untouched

Run it (Windows, from this folder)
----------------------------------
1. pip install -r backend/requirements.txt
2. Copy model.pth into the backend folder.
3. python backend/server.py
4. Open http://localhost:8000 in the browser.
   Press "Run Demo" or pick a sample image / upload an image and press
   "Generate AI Prediction".

Without model.pth the page still works in DEMO MODE (clearly labelled placeholder
predictions). The top-left status shows ONLINE when the real model is loaded.

Check that the backend works:   python backend/tests/smoke_test.py
Check with the real model:      python backend/tests/smoke_test.py --real

What is real and what is simulated
----------------------------------
REAL       intensity class, confidence, class probabilities (EfficientNet-B0, ~60% accuracy
           on 372 unseen test images)
SIMULATED  track outlook, wind outlook, pressure, risk index, rapid-intensification flag
           (copied from similar historical storms / simple rules)
The page shows tags (MODEL / SIMULATED) on every value so judges are never misled.

Internet: the map and chart libraries and fonts load from CDNs. Without internet the page
falls back to a simple track plot and chart, so the demo still works.
