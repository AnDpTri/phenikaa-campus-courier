# V15 role-aware synthetic generator — 2,000-example pilot audit

Date: 2026-10-09
Status: PASS implemented semantic-role contracts on the 2,000-example pilot;
NOT YET CERTIFIED as a full-quality replacement for the original 40K.

## Source of improvements
Conceptual structure from user-supplied robot_hard_training_dataset(1).zip
(text_train examples and label ontology); no row text or validation split was
copied into the generator. Do not directly train the V2 schema on ZIP's
relative farthest labels, extra place types, and tri-state flags without
faithful mapping.

## New semantics introduced compared with V11

- `former_destination` with explicit supersession and genuine final goal
- retracted prior pickup stop, either replaced by an active via or resulting in via=null
- 6 distinct negative mention roles: avoid, not_stop, closed, historical, left, not_related
- can pick up a supplement item different from the main delivered parcel
- explicit urgency/fragility updates and some cases with no flag mention
- no assumption that the presence of "trước" fully determines active via
- no assumption that "lấy" implies active via
- goal/via spatial modes restricted to currently supported V2 label ontology
- for map-only goals, negative mentions limited to historical or unrelated
  places, rather than banning a possible selected final map node
- fixed noun-noun "bưu kiện thùng..." and unnatural "ngoài kiện bưu kiện"
- fragile statements avoid incompatible cargo-specific extra details

## V15 actual pilot

File: pilot_2000_v15_role_semantics.jsonl
Source script: role_semantic_generator_v12.py
Reproducible runner: run_semantic_generator_v15.py
Provenance: pilot_2000_v15_provenance.jsonl
Generator report: pilot_2000_v15_generation_report.json
Independent verifier: run_independent_role_audit_v15.py
Audit: independent_semantic_audit_v15_2000.json
Selected stratified samples: stratified_semantic_review_v15.jsonl

2,000 entirely newly generated records, new random seed.
- true via: 940/2000 (47.0%)
- former destination explicitly superseded: 444/2000 (22.2%)
- former destination superseded + true via: 207/2000 (10.35%)
- former pickup canceled: 271/2000 (13.55%)
- via present without 'trước': 562
- via absent despite 'lấy' appearing: 461
- separate pickup item: 484
- role-tagged negative-place mentions: 907 occurrences
- accented text: 1172/2000 (58.6%)
- median 75 tokens, P90 104 tokens, max 160
- SHA256: b4d6184cb91c603099b106b25fe035ce685622ee28df6693dc1fc32ed71ac444

The independent rule audit reported ISSUES {} after checking all 2,000 rows,
including readable role-provenance semantics, V2/V4 target schema, presence
of positive flag evidence, valid revocation markers, safe role-to-target
separation, length and known concatenation errors. Sample rows were also
read at distributed locations and found no obvious role reversal.

LIMITATIONS:
- Audit is structural and partial manual reading, not exhaustive semantic
  annotation. No quantified claim that 100% of rows are correct.
- This is ONLY a 2K pilot. A new 40K has not yet been built from this version.
- P90 sentence length remains longer than the reference 40K (104 vs 85).
- Coverage of background distractors and naturalness still requires a
  measured comparison before calling it fully equivalent to the original 40K.
- No model training, no changes to source, checkpoints or Kaggle submission.
- All old datasets including incoming_40000.jsonl and V11 remain untouched.

Next quality gate: fresh expanded generation and blind semantic review;
only then 40K final and controlled training.
