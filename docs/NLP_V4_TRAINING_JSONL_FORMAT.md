# NLP v4 training JSONL format

The trainer accepts one JSON object per line. Do not wrap records in a JSON array.
UTF-8 is required. Keep the text in normal Vietnamese or accent-folded Vietnamese;
the current synthetic corpus is lowercase ASCII with deliberate case and typo noise.

## Required object

```json
{
  "id": "extra_000001",
  "text": "nho robot mang goi hang toi phong hoc",
  "goal": {"type": "lecture", "ref": null, "anchor": null},
  "via": null,
  "urgent": false,
  "fragile": false
}
```

`goal` is required. `via` is either `null` or another target object. A target object
has exactly these semantic fields:

```json
{"type": "library", "ref": null, "anchor": null}
```

Allowed `type` values are `library`, `dorm`, `sports`, `clinic`, `canteen`, `parking`,
`lecture`, `lab`, `office`, and `gate`. Use `null` for map-only targets.

Allowed `ref` values are `null`, `north`, `south`, `west`, `east`, `near`, `far`,
`anchor_near`, `north_most`, `south_most`, `west_most`, and `east_most`.

Rules for labels:

- A named destination uses `type` and `ref: null`, with `anchor: null`.
- `near` and `far` require both a `type` and an `anchor` type.
- `anchor_near` means the closest location to `anchor`; use `type: null`.
- The four `*_most` values mean a map extreme; use `type: null` and `anchor: null`.
- Directional `north`, `south`, `east`, and `west` use a `type` and no anchor.
- `urgent` and `fragile` must be JSON booleans, not strings such as `"false"`.
- `text` must support the labels by itself. Do not include a hidden answer or a test-only identifier.
- A via target is an intermediate stop that must be visited before the final goal; it is not the final destination.

## JSONL example with via and spatial labels

```json
{"id":"extra_000002","text":"truoc tien ghe nha an lay hop thuoc roi giao toi phong y te","goal":{"type":"clinic","ref":null,"anchor":null},"via":{"type":"canteen","ref":null,"anchor":null},"urgent":false,"fragile":false}
{"id":"extra_000003","text":"mang bo tai lieu toi khu gan ky tuc xa nhat","goal":{"type":null,"ref":"anchor_near","anchor":"dorm"},"via":null,"urgent":true,"fragile":false}
{"id":"extra_000004","text":"giao thung carton den vi tri cuc bac cua ban do","goal":{"type":null,"ref":"north_most","anchor":null},"via":null,"urgent":false,"fragile":true}
```

## Local validation

Use the same loader before adding a file to the training run:

```powershell
$env:PYTHONPATH = "src"
py -3.12 -c "from pathlib import Path; from scripts.nlp.train_neural_parser import labelled_prebuilt; rows=labelled_prebuilt(Path('your_file.jsonl')); print(len(rows))"
```

The loader checks JSON structure while reading. It does not prove that the wording
matches the labels, so sample and review newly authored rows before training.
