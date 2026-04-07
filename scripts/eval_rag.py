from __future__ import annotations

"""
Simple evaluation runner:
- Sends questions to /ask-question
- Measures latency
- Computes semantic similarity between predicted vs expected (via embeddings endpoint-free local call is not available)

For now, this script focuses on latency and basic exact match; extend as needed.
"""

import time
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_contains: list[str]


CASES = [
    EvalCase(question="What is this document about?", expected_contains=[]),
    EvalCase(question="List the key points.", expected_contains=[]),
]


def main() -> None:
    base = "http://localhost:8000"
    user_id = "eval"
    session_id = None

    results = []
    for c in CASES:
        t0 = time.time()
        r = requests.post(
            f"{base}/ask-question",
            json={"user_id": user_id, "session_id": session_id, "question": c.question, "doc_ids": None},
            timeout=180,
        )
        dt_s = time.time() - t0
        r.raise_for_status()
        data = r.json()
        session_id = data["session_id"]
        ans = data["answer"]

        ok = True
        for needle in c.expected_contains:
            if needle.lower() not in ans.lower():
                ok = False
                break
        results.append({"question": c.question, "latency_s": dt_s, "pass": ok})
        print(f"- q={c.question!r} latency={dt_s:.2f}s pass={ok}")

    passed = sum(1 for r in results if r["pass"])
    print(f"\nPassed {passed}/{len(results)}")


if __name__ == "__main__":
    main()

