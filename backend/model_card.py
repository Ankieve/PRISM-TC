"""
Model Transparency panel data (SIH26070 / PRISM-TC).

Everything in MODEL_CARD below is copied from CLAUDE.md and the team's own
test run of the shipped classifier - nothing here is invented for the demo.
The goal of this module is to make those numbers and caveats available to
the frontend as structured JSON instead of leaving them only in a markdown
file nobody in the room will open. If a number below is ever updated after
retraining, update it here AND in CLAUDE.md's "Pitch numbers" section so the
two never drift apart.

This endpoint (GET /api/model/transparency) reports facts about the model
that was trained. It is intentionally static/backend-only: it does not
change per-request the way /api/predict does.
"""

MODEL_CARD = {
    "model": {
        "architecture": "Ensemble of 2 EfficientNet-B0 models (via timm), each a "
                         "wind-speed REGRESSION head (1 output, kt) rather than a "
                         "direct 5-class classifier. Outputs are averaged across "
                         "the 2 members and across 4 rotations of the same image "
                         "(test-time augmentation), then converted to an IMD "
                         "class via fixed wind thresholds (34/48/64/90 kt). "
                         "Replaces the earlier v2 single-model direct classifier.",
        "input": "Single infrared satellite image chip, resized to 224x224, RGB "
                 "tensor WITH standard ImageNet normalization (v2 was trained "
                 "WITHOUT normalization - this is a real pipeline change between "
                 "versions, not an oversight).",
        "training_data": "TCIR dataset only, same images as v2 (no MOSDAC data is "
                          "in this training set yet - that collection/build "
                          "pipeline is still in progress, see CLAUDE.md). What "
                          "changed is the split: a storm-wise (grouped by "
                          "storm_id) train/val/test split, so no storm's images "
                          "can appear in both train and test. v2's split allowed "
                          "that, which is why v2 scored 97.2% on its own \"train\" "
                          "storms but only 54.8% on unseen ones - a 42.4-point "
                          "gap - when re-measured on this same storm-wise split.",
    },

    "classes": [
        "Depression",
        "Cyclonic Storm",
        "Severe Cyclonic Storm",
        "Very Severe Cyclonic Storm",
        "Extremely Severe/Super Cyclone",
    ],

    "metrics": {
        "test_set_size": 520,
        "test_set_storms": 12,
        "test_set_note": "12 storms held out entirely (none of their images seen "
                          "in training) - a stricter test than v2's original "
                          "372-image split, which did not control for this.",
        "accuracy_exact_pct": 53.7,
        "accuracy_pct": 53.7,  # alias: the frontend's Model Transparency panel
                               # (app.js loadModelTransparency) reads this exact
                               # key name. Kept in sync with accuracy_exact_pct
                               # above - update both together.
        "accuracy_within_one_class_pct": 93.1,
        "macro_f1": 0.487,
        "mean_absolute_error_kt": 10.2,
        "average_error_severity_levels": 0.535,  # frontend key (app.js
            # loadModelTransparency reads m.average_error_severity_levels).
            # Computed from a per-image export cell added to MODAK.ipynb
            # right after cell 21 - see analytics/inputs/reported_metrics.json
            # for the full methodology note. Update both together.
        "expected_calibration_error_pct": 7.61,  # frontend key (app.js reads
            # m.expected_calibration_error_pct). Computed from the same
            # per-image export, 10 equal-width bins on the model's DERIVED
            # (Normal-CDF, not softmax) confidence - see known_failure_modes
            # below. Not directly comparable to v2's ~27% ECE (different kind
            # of confidence), so don't present this as "4x better than v2."
        "train_test_gap_points": 6.0,
        "v2_comparison_same_test_set": {
            "accuracy_exact_pct": 54.8,
            "accuracy_within_one_class_pct": 87.9,
            "macro_f1": 0.453,
            "train_test_gap_points": 42.4,
            "note": "v2 re-evaluated on this same storm-wise 520-image test for "
                    "a fair comparison - a different (harder) test than the "
                    "60.5% v2 was originally reported on, so the two v2 numbers "
                    "are not the same measurement.",
        },
        "live_biparjoy_validation": {
            "source": "36 real INSAT-3DR frames of Cyclone Biparjoy (Jun 2023), "
                       "IBTrACS ground truth - the only live-satellite check "
                       "either model has had.",
            "ensemble_accuracy_exact_pct": 75.0,
            "ensemble_accuracy_within_one_class_pct": 100.0,
            "ensemble_mae_kt": 4.9,
            "v2_accuracy_exact_pct": 83.3,
            "v2_accuracy_within_one_class_pct": 94.4,
            "honest_caveat": "v2 actually scored HIGHER exact accuracy than the "
                              "ensemble on this one live storm, though the "
                              "ensemble won on within-one-class and adds an MAE "
                              "figure v2 cannot produce. This is not hidden: v2's "
                              "strong Biparjoy number may itself reflect the "
                              "overfitting the storm-wise test exposed, so it is "
                              "not a reason to prefer v2 on a new, unseen storm - "
                              "but the honest claim for the ensemble is 'more "
                              "consistent across many unseen storms,' not "
                              "'better on every number.'",
        },
        "note": "Measured on a held-out TCIR storm-wise test split plus one live "
                "satellite validation (Biparjoy). Not yet re-measured against a "
                "broader set of live satellite imagery.",
    },

    "known_failure_modes": [
        "Confidence shown for this model is NOT a measured softmax probability: "
        "it is derived from the wind estimate and the model's own measured test "
        "MAE (10.2 kt) via a Normal-distribution assumption (see predict.py). "
        "Its Expected Calibration Error, measured the same way (10 bins, on the "
        "520-image storm-wise test set), is ~7.6% - lower than v2's ~27%, but "
        "the two are not directly comparable since v2's ECE calibrated a real "
        "softmax and this one calibrates a derived Normal-distribution estimate. "
        "Read it as 'this derived confidence tracks accuracy reasonably well,' "
        "not as a proven 4x calibration improvement over v2.",
        "Single infrared channel only - no visible, water-vapour, or microwave channel, "
        "unlike operational Dvorak-style intensity estimation.",
        "Still trained on TCIR only - no MOSDAC/INSAT data has been added to "
        "training yet, only used for the separate Biparjoy live-validation check.",
        "Trained on pre-cropped, storm-centred chips. Does not locate or detect a "
        "storm inside a raw satellite frame - it only classifies an image someone "
        "(or a separate cropping step) has already centred.",
        "No temporal/sequence model: each classification is a single independent image, "
        "not a track of a storm's evolution over time.",
        "Exact-match accuracy (53.7%) is modest in absolute terms; within-one-class "
        "(93.1%) is the more reliable headline number for a 5-class problem with "
        "adjacent classes this close in wind speed.",
    ],

    "honesty_notes": [
        "Track, wind outlook, pressure, risk index and the analog-based "
        "rapid-intensification flag are historical-analog and rule-based methods, "
        "not machine-learning forecasts - see backend/logic.py.",
        "SST and wind shear now feed a separate, clearly-labelled rule-based "
        "environment heuristic (not the AI classifier) - see backend/logic.py's "
        "environment_favors_intensification(). Humidity and vorticity remain "
        "accepted-but-unused, reserved for a possible future ERA5 fusion.",
    ],
}


def get_model_card():
    """Return a fresh copy (so callers can't mutate the shared dict) plus a
    live 'calibration' section.

    As of the v3 wind-regression ensemble, temperature scaling (calibration.py
    / backend/temperature.json) no longer applies: that mechanism rescales
    softmax logits from a direct classifier, and this model has no logits -
    its confidence is already derived differently (see predict.py and the
    'known_failure_modes' note above). A temperature.json left over from
    fitting it against the old v2 model is NOT applied to v3 predictions -
    reporting it as if it still were would be exactly the kind of misleading
    claim this project's honesty rules exist to prevent. This still imports
    calibration.py (needs no torch) purely to report that fact accurately."""
    import copy

    from calibration import load_temperature

    card = copy.deepcopy(MODEL_CARD)
    _temperature, fitted = load_temperature()
    if fitted:
        card["calibration"] = {
            "fitted": False,
            "temperature": 1.0,
            "note": "backend/temperature.json exists (fitted against the old v2 "
                    "model) but is NOT used by the current v3 ensemble - "
                    "temperature scaling rescales softmax logits, and this "
                    "model has no logits (see 'known_failure_modes' above for "
                    "how its confidence is actually derived).",
        }
    else:
        card["calibration"] = {
            "fitted": False,
            "temperature": 1.0,
            "note": "Not applicable to the current v3 ensemble - it has no softmax "
                    "logits to rescale (see 'known_failure_modes' above for how its "
                    "confidence is actually derived). backend/fit_calibration.py "
                    "still targets the old v2 architecture and has not been "
                    "adapted for v3; do not run it against v3 checkpoints.",
        }
    return card
