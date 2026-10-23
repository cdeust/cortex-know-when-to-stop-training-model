"""Exercise the installed public heuristic API, with no trained-model claim.

Sources: classifier.py public API and heuristic.py token-overlap contract.
The explicit fallback mode avoids model resolution and downloads.
"""

from pathlib import Path
import sys

import cortex_beam_abstain
from cortex_beam_abstain import AbstentionClassifier


def main() -> None:
    installed = Path(cortex_beam_abstain.__file__).resolve()
    assert installed.is_relative_to(Path(sys.prefix).resolve()), installed
    classifier = AbstentionClassifier(use_heuristic=True)
    query = "alpha beta"
    relevant = "alpha beta"
    irrelevant = "gamma delta"
    # Full overlap and no overlap are the endpoints of the documented score.
    assert classifier.predict(query, relevant) == 1.0
    assert classifier.predict(query, irrelevant) == 0.0
    pairs = [(query, relevant), (query, irrelevant)]
    assert classifier.predict_batch(pairs) == [1.0, 0.0]
    assert classifier.predict_batch([]) == []
    assert classifier.should_abstain(query, []) is True
    assert classifier.should_abstain(query, [irrelevant]) is True
    assert classifier.should_abstain(query, [irrelevant, relevant]) is False
    # The threshold comparison is strict: equality must remain relevant.
    assert classifier.should_abstain(query, [relevant], threshold=1.0) is False
    print(f"Installed package: {installed}")
    print("Heuristic API: single, batch, empty results and abstention passed")


if __name__ == "__main__":
    main()
