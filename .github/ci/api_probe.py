"""Exercise the public heuristic fallback API, without a model download.

Sources: tests/test_classifier.py and classifier.py predict_batch/should_abstain.
No trained-model quality or ONNX inference claim follows from this probe.
"""

from cortex_beam_abstain.classifier import AbstentionClassifier


def main() -> None:
    classifier = AbstentionClassifier(use_heuristic=True)
    # Existing examples from test_classifier.py, not a new quality benchmark.
    pairs = [
        (
            "What language do they use?",
            "The team decided to use TypeScript for the frontend project.",
        ),
        (
            "What recipe did they discuss?",
            "We fixed the authentication bug in the API middleware.",
        ),
    ]
    scores = classifier.predict_batch(pairs)
    assert scores == [classifier.predict(query, passage) for query, passage in pairs]
    assert classifier.should_abstain("query", []) is True
    for (query, passage), score in zip(pairs, scores, strict=True):
        assert classifier.should_abstain(query, [passage]) == (
            score < classifier.threshold
        )
    print("Public heuristic API: predict, batch and abstention agree")


if __name__ == "__main__":
    main()
