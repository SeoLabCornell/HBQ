from __future__ import annotations

import re
from typing import Any

import datasets
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed


def _target_answer(doc: dict[str, Any]) -> str:
    answer = doc.get("answer")
    if answer is not None and str(answer).strip():
        return str(answer)

    solution = doc.get("solution", "")
    boxed = last_boxed_only_string(solution)
    if boxed is not None:
        return remove_boxed(boxed)
    return str(solution)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict[str, Any]) -> dict[str, str]:
        return {
            "problem": str(doc["problem"]),
            "solution": str(doc.get("solution", "")),
            "answer": _target_answer(doc),
        }

    return dataset.map(_process_doc)


def _extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    boxed = text[idx:]
    try:
        return remove_boxed(last_boxed_only_string(boxed))
    except Exception:
        return None


def _extract_answer(text: str) -> str:
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    final_patterns = [
        r"(?i)final answer is\s*[:\-]?\s*(.+)",
        r"(?i)the answer is\s*[:\-]?\s*(.+)",
        r"(?i)answer\s*[:\-]\s*(.+)",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text)
        if matches:
            candidate = matches[-1].strip().split("\n")[0].strip()
            return candidate.strip(" .")

    dollar_spans = re.findall(r"\$([^$]+)\$", text)
    if dollar_spans:
        return dollar_spans[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, int]:
    prediction = _extract_answer(results[0])
    return {"exact_match": int(is_equiv(prediction, doc["answer"]))}



def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] | None = None):
    # Do not import lm_eval.tasks.humaneval.utils here: this lm-eval version
    # executes code_eval at import time. Load the metric lazily only when the
    # HumanEval task actually computes pass@k.
    import evaluate as hf_evaluate

    assert k is not None
    if isinstance(k, int):
        k = [k]
    compute = hf_evaluate.load("code_eval")
    return compute.compute(references=references, predictions=predictions, k=k)[0]


def _clean_humaneval_completion(text: str) -> str:
    markers = [
        "\nWait,",
        "\nWait",
        "\nOkay,",
        "\nOkay",
        "\nTo solve",
        "\nThis function",
        "\nThe function",
        "\n```",
        "\n</think>",
        "\n<｜",
    ]
    cut = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    cleaned = text[:cut].rstrip()
    return cleaned + "\n" if cleaned else text


def build_predictions_clean(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + _clean_humaneval_completion(resp) for resp in resp_list]
        for resp_list, doc in zip(resps, docs)
    ]
