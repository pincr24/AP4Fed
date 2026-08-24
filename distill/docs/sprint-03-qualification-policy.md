# Sprint 03 qualification-only policy

This milestone connects the real Sprint 01 CONFOLD output to the closed-loop
AP4Fed controller without presenting it as a validated policy. The
implementation and deterministic checks are review-ready. The paired Docker
trajectories and their state-parity evidence remain pending.

## Rule decision

The inputs are the three frozen Sprint 01 exploratory rule reports and the
frozen eighty-row warm-state dataset under
`distill/data/paper_archive_fashionmnist_cnn16k_fewshot_deepseek/`. Their exact
hashes are pinned by `qualification_rules.py`.

No learned threshold, condition, confidence, or rule is tuned for Sprint 03.
The only action completion is Client Selector `selection_value=3`, chosen before
the run from the declared `[5, 5, 5, 3, 3]` CPU layout. It is still passed
through AP4Fed's existing selector safety check.

CONFOLD's reports are ordered decision lists and include one nested exception.
The runtime evaluates independent unordered rules. The producer therefore
performs a fixed semantic translation:

1. preserve each earlier rule's effective region;
2. express later default rules only over regions not claimed earlier;
3. flatten the nested exception by Boolean expansion;
4. split disjunctions into separate rules with the same action.

The result contains six Client Selector rules, six Message Compressor rules,
and five HDH rules. A deterministic test compares the actions of the source
ordered lists and translated rules on all eighty frozen warm states. It also
requires every translated state to be unambiguous.

The artifact is labelled `artifact_kind=qualification_only` and
`intended_use=closed_loop_qualification`. It records training coverage and
precision, not held-out validation. The independently validated artifact loader
still requires support and Wilson-bound evidence and does not accept this
format.

## Runtime behavior

The opt-in `Distilled Policy` has three modes:

- `always_defer` captures the state and always asks the declared teacher;
- `shadow` evaluates the rules and records their proposal, but still applies
  the teacher action;
- `active` applies a rule proposal only when all three heads decide. Cold
  start, no rule, or conflicting actions defer the whole decision to the
  teacher.

The runtime verifies the artifact's exact file hash, qualification designation,
source provenance, and exact AG News configuration before the run begins. Each
trace records the artifact ID/hash, fired rules, deferral reason, teacher status,
pre-guardrail action, applied action, guardrail result, and controller time.
Teacher prompts and responses are retained only when the teacher is queried.

The Docker server mounts `distill/` read-only at `/distill`; Local and Docker
continue to use byte-identical adaptation hooks.

## Frozen qualification configuration

`distill/configs/sprint03_agnews_base.json` preserves the agentic-paper AG News
/ MLP configuration: ten rounds; five clients; CPU `[5,5,5,3,3]`; RAM 2;
the archived IID/non-IID, Same/New Data, and delay layout; delay 20--50 seconds
on clients 3 and 4; non-IID alpha 0.9; and partition seed `2764335072`. The
three controlled patterns start OFF and Client Selector uses threshold 3.

The preparation command validates these fields and writes a new file rather
than replacing an existing run configuration. After the implementation has
been reviewed and committed, build the artifact from that clean checkout:

```sh
python3 distill/build_sprint01_qualification_rules.py \
  --producing-code-sha "$(git rev-parse HEAD)" \
  --output distill/artifacts/sprint03/sprint01-qualification.json
```

Prepare the paired configurations with different immutable run IDs:

```sh
python3 distill/prepare_sprint03_run.py \
  --mode always_defer \
  --run-id sprint03/agnews/control/r1 \
  --teacher-model-digest <full-model-sha256> \
  --output <new-control-config.json>

python3 distill/prepare_sprint03_run.py \
  --mode active \
  --run-id sprint03/agnews/active/r1 \
  --teacher-model-digest <full-model-sha256> \
  --artifact distill/artifacts/sprint03/sprint01-qualification.json \
  --output <new-active-config.json>
```

The artifact builder refuses a dirty checkout or a producing SHA different
from `HEAD`. The configuration producer computes the artifact hash itself and
requires the artifact to remain below `distill/` so the same relative path is
available in Docker.

## Checks and interpretation

From the AP4Fed repository root:

```sh
python3 -m unittest distill.tests.test_qualification_rules -v
python3 -m unittest discover -s distill/tests -v
```

The focused tests cover pinned source hashes, exact configuration, mechanical
source-rule equivalence, qualification/validation separation, cold-start
deferral, unambiguous warm-state dispatch, shadow behavior, selector safety,
rule-only traces, and paired configuration preparation.

These checks establish deterministic implementation behavior only. A poor
closed-loop F1 or runtime result will not fail Sprint 03, and a good result will
not validate the rules. The sprint result requires the reviewed paired Docker
runs, one immutable trace per decision, offline/live state parity, and a
provenance-complete run manifest.
