# Errata for semantic_audit_v7_manual_171_20261009.md

The semantic audit report is immutable under the user's CREATE-NEW-ONLY rule.
The following observation in the "Positive examples" list needs correction:

- **REMOVE positive example `specnew_001312`.** The reference to goal west-most
  and via clinic-near-library was mistakenly carried over from a previous pilot.
  In V7, `specnew_001312` is actually `goal=parking/far/dorm`, `via=null`.
- **USE `specnew_001311` instead:** V7 has `goal=dorm/named` and
  `via=gate/named`; its explicit instructions correctly place pickup before
  final delivery. This is present in the 171-row stratified inspection set.

This erratum does not affect the quantified audit statistics or the final
**FAIL semantic-parity** conclusion. Existing audit and data files were not altered.
