# Treeval

## Code examples

```Python
from treeval import treeval

import evaluate

sacrebleu_metric = evaluate.load("sacrebleu")
f1_metric = evaluate.load("f1")
exact_match_metric = evaluate.load("exact_match")

prediction = {
    "music_piece_name": "Die With A Smile",
    "author_composer_name": "Lady Gaga and Bruno Mars",
    "total_number_of_bars": 10,
    "shouldnt_be_here": True,
    "lyrics": [
        "I just woke up from a dream where you and I had to say bye-bye.",
        "And I don't know what it all meant, but since I survived I realized.",
        "Wherever you go, that's where I'll follow.",
    ],
}
reference = {
    "music_piece_name": "Die With A Smile",
    "author_composer_name": "Lady Gaga, Bruno Mars",
    "total_number_of_bars": 8,
    "missing_field": False,
    "lyrics": [
        "I just woke up from a dream, where you and I had to say good-bye.",
        "And I don't know what it all means, but since I survived I realized.",
        "Wherever you go, that's where I'll follow.",
    ],
}
schema = {
    "music_piece_name": "string",
    "author_composer_name": "string",
    "total_number_of_bars": "integer",
    "lyrics": ["string"],
}
metrics = {
    "sacrebleu": {
        "callable": sacrebleu_metric,
        "fn_score": 0,
        "fp_score": 0
    },
    "f1": {
        "callable": f1_metric,
        "fn_score": 0,
        "fp_score": 0
    },
    "exact_match": {
        "callable": exact_match_metric,
        "fn_score": 0,
        "fp_score": 0
    },
}
leaves_metrics = {
    "music_piece_name": ["sacrebleu", "exact_match"],
    "author_composer_name": ["sacrebleu", "exact_match"],
    "total_number_of_bars": ["exact_match"],
}
types_metrics = {
    "integer": ["exact_match"],
    "string": ["sacrebleu"],  # will be run for "lyrics" (not provided in nodes_metrics)
}

result = treeval([prediction], [reference], schema, metrics)
```
