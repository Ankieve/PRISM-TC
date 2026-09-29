# VAYUVEGA (PRISM-TC) — Roadmap & Vision

**SIH26070** · This document exists so the ambition of this project is on record without
overstating what is built today. Read it alongside `CLAUDE.md` (how the current system works)
and `PRISM-TC_SIH_Gap_Analysis.md` (an honest audit of today's limitations). Everything in
Phase 1 below is real and running. Everything after it is roadmap, not a claim.

## Positioning

Not "another cyclone-tracking website." The intended end state:

> Don't just predict the cyclone. Understand it, explain it, quantify uncertainty, predict its
> consequences, identify who may be affected, and help decision-makers respond.

*Observe → Detect → Forecast → Explain → Quantify uncertainty → Predict impacts → Prioritize
risk → Support decisions → Learn from outcomes.*

A platform at that scale needs a meteorology team, an oceanography team, GIS engineers,
disaster-management specialists, a production ML pipeline, and data-sharing arrangements with
IMD/INCOIS/ISRO — it is a multi-year, multi-team build, not a hackathon deliverable. This
document breaks it into phases so the current prototype's place in that arc is clear, and so
later teams (or a longer-running version of this one) have a concrete plan instead of a wish
list.

---

## Phase 1 — What actually exists today (built, tested, running)

- A trained EfficientNet-B0 classifier for the 5 IMD intensity classes, ~60% accuracy /
  macro-F1 0.49 on 372 held-out TCIR images, honestly reported with its calibration error
  (~27% ECE) and known failure modes — see the in-app Model Transparency panel.
- A historical-analog track/wind outlook and a small set of documented meteorological
  rules (SST + wind-shear intensification check), clearly tagged SIMULATED/rule-based, never
  presented as ML.
- A near-real-time satellite feed (IMD/INSAT-3D IR1), polled and honestly labelled
  LIVE/STALE/ERROR — not fused with the classifier, by design (train/inference mismatch is
  disclosed, not hidden).
- An out-of-distribution input guard, a backtest mode against real IBTrACS storms, Grad-CAM
  and temperature-scaling calibration (code-complete; calibration/Grad-CAM need the operator's
  own machine with torch + the real model weights to fit/verify).
- A References & Glossary panel citing the actual papers and datasets used (TCIR, IBTrACS,
  EfficientNet, Guo et al. 2017, Selvaraju et al. 2017, IMD bands).

This is a single-cyclone, single-source, request-driven prototype: one image in, one
classification out, with a simulated outlook alongside it. It does not run continuously, has no
database, no GIS layer, and no multi-model ensemble. That gap is intentional context for
everything below.

---

## Phase 2 — Real multi-source AI forecasting

**Adds:** track/intensity/rainfall/wind models trained on more than one data source; a
rapid-intensification detector using validated features (SST, ocean heat content, wind shear,
moisture); a calibrated uncertainty estimate instead of a single point forecast.

**Why:** this is the most direct answer to the problem statement's "multi-source satellite
data" and "prediction" language — today's system is single-source and its "prediction" is a
historical analog, not learned. This phase is where that stops being true.

**Needs:** ERA5 or equivalent reanalysis (see the ERA5 module shipped with this update),
INSAT/Himawari-derived cyclone-centred imagery at scale, and enough labelled storms to
train/validate a track and intensity model separately from the classifier. Feasible for a
longer student project; not feasible to fabricate honestly in a hackathon timeline.

**Validation:** held-out storms, mean track/position error in km, intensity MAE/RMSE, compared
against a persistence/climatology baseline (the backtest mode already in this app is a working
example of exactly this kind of evaluation, just for the rule-based method).

## Phase 3 — Ocean state and impact

**Adds:** SST, ocean heat content, wave/swell height (from public sources: NOAA OISST, INCOIS
buoys/wave data where available); a storm-surge estimate combining intensity, pressure, wind
field, bathymetry and tide; a first-pass coastal-flooding overlay against elevation data; district-level
population and infrastructure exposure counts.

**Why:** this is where "prediction" starts to mean something to a person on the coast, not just
a probability over 5 classes.

**Needs:** bathymetry (GEBCO, public), a population raster (WorldPop or census-derived, public),
OSM infrastructure layers (public) — all obtainable without a government data-sharing
agreement, which makes this phase realistic to prototype even outside a large team, unlike
Phases 5 and 7.

**Limitation to state plainly if this is ever built:** a storm-surge estimate without validated
bathymetry and a real hydrodynamic model (e.g. ADCIRC, SLOSH) is a rough order-of-magnitude
figure, not an operational surge forecast — it should stay labelled as such.

## Phase 4 — Digital twin, ensembles, and explainability

**Adds:** a live "current state → observed evolution → forecast state → potential impact" view
that updates as new observations arrive and keeps history for replay; more than one model
(e.g. a gradient-boosted model alongside the CNN) so **disagreement between models** can be
shown honestly instead of a single confident number; SHAP/feature-importance style explanations
for why a prediction was made; an automatic "why did the forecast change?" note when a new
observation shifts the outlook.

**Why:** disagreement-hiding is explicitly called out as a failure mode to avoid in the source
document this roadmap is built from — showing "Model A says X, Model B says Y, here is the
resulting uncertainty" is more credible to an evaluator than false precision.

**Needs:** at least two independently-trained models of comparable quality (not just one model
run twice), and a real observation stream to replay against — both are Phase-2/3 prerequisites.

## Phase 5 — Disaster decision support

**Adds:** evacuation-priority ranking by hazard severity, population exposure and shelter
accessibility; a shelter database with capacity/status/routes; a route engine that recalculates
when a road becomes unavailable; a "what-if" scenario simulator (e.g. "landfall shifts 100km
north") clearly labelled **SCENARIO SIMULATION — NOT AN OFFICIAL FORECAST**.

**Why:** this is the layer that turns a forecast into something a district disaster-management
officer can act on.

**Needs:** real shelter location/capacity data and a real road network with live
closure/flood information — this is the phase most dependent on government or NGO data
partnerships (e.g. state disaster management authorities, NDMA), and is not realistic to
prototype convincingly without them; a fabricated shelter/route database would be actively
misleading in a disaster-support tool, more so than in any other part of this system.

## Phase 6 — GenAI command layer

**Adds:** a conversational assistant over the platform's own structured data ("which districts
have the highest exposure right now", "why did the track shift"), natural-language map control
("show storm surge", "show shelters"), and automated situation-report generation.

**Why:** lowers the barrier for a non-specialist (a citizen, a junior officer) to get a
correct, sourced answer instead of reading raw panels.

**Needs:** everything above to already exist and be queryable — a GenAI layer with no real
data behind it is a chatbot that plausibly invents cyclone facts, which is close to the worst
failure mode a disaster tool can have. This phase strictly follows, never leads.

## Phase 7 — Production hardening

**Adds:** authentication/rate limiting, model versioning and an audit trail (every forecast
traceable to a model version + input data + timestamp), automated post-event evaluation
(prediction vs. actual, logged per storm), monitoring, backups, failover.

**Why:** the difference between a prototype and something a disaster-management office could
actually rely on.

**Needs:** this is standard software/ML-ops work once Phases 2-5 exist; it is the least
research-risky phase and the most pure-engineering one.

---

## What this roadmap deliberately leaves out of any near-term plan

A single "composite impact score" combining wind/rain/surge/flood/population/infrastructure
into one number per district is listed in the source vision document, but a combined score like
that hides exactly the kind of uncertainty this whole project's honesty rules exist to surface —
if built, each component should stay separately visible, with the composite shown as one
more, clearly-labelled derived view, not the headline number.

## How to use this in the pitch

Say what Phase 1 actually does, in plain terms, before naming the vision — a judge who checks
and finds a 5-class image classifier after hearing "AI-powered real-time disaster intelligence
platform" will trust the team less, not more. Frame it as: *"Here is a working, honestly-evaluated
piece of Phase 1. Here is the phased plan for the platform it plugs into, and here is exactly
what each later phase needs that we don't have yet."* That is a stronger story than claiming
more than what a judge can click through and verify in the demo.
