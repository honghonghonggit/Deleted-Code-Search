"""분류 정확도 평가 - 예측 파일을 확정 라벨의 한 split 과 맞춰 본다 (#86, CHARTER §10.2). 담당: 희수

무엇을:
    우리 방식(`classify.classifier`)과 기준선 A·B(`classify.baseline_keyword`,
    `classify.baseline_llm`)의 예측 파일은 모두 `record_id` + `predicted_label` 로 시작한다
    (`classify.baselines`). 그것을 `datasets/labeled_500.jsonl` 의 `final.reason_label` 과 비교해
    §10.2 지표를 낸다 - 정확도, 클래스별 F1, 혼동 행렬, 근거 등급별 정확도. 세 방식을 **같은
    코드**로 재야 차이가 계산 탓이 아니다 (#87).

    게이트 2 판정(기준·H)은 하지 않는다. 그건 #92 다. 여기는 숫자만 낸다.

"모두 UNK" 를 같이 내는 이유:
    split 마다 UNK 비율이 달라(학습 45% · 검증 39%) 정확도만 보면 "UNK 만 답해도 나오는 만큼" 을
    가늠할 수 없다. 그 바닥을 같은 표에 둔다.

실행:
    python -m classify.evaluate predictions.jsonl --split val
    python -m classify.evaluate predictions.jsonl --split val --json docs/reports/x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from classify.baselines import UNKNOWN_LABEL
from classify.labels import EVIDENCE_GRADES, MERGED_FILENAME, REASON_LABELS, read_jsonl

DEFAULT_LABELS = Path("datasets") / MERGED_FILENAME
SPLITS = ("train", "val", "test")


def load_gold(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, Mapping[str, Any]]:
    """병합 파일 → 그 split 의 `{record_id: final}`. `final` 이 확정되지 않은 건이 있으면 거부한다.

    확정되지 않은 건을 빼고 재면 분모가 split 마다 달라지고, 한쪽 라벨로 채우면 그 사람의 판단을
    정답으로 쓰는 셈이다. 500건은 모두 확정됐으므로(#166) 여기 걸리면 파일이 잘못된 것이다.
    """
    gold: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("split") != split:
            continue
        final = row.get("final") or {}
        if final.get("reason_label") not in REASON_LABELS:
            raise ValueError(f"{row.get('record_id')}: final.reason_label 이 확정되지 않았다")
        gold[str(row["record_id"])] = final
    if not gold:
        raise ValueError(f"split={split} 인 레코드가 없다")
    return gold


def evaluate(gold: Mapping[str, Mapping[str, Any]], predicted: Mapping[str, str]) -> dict[str, Any]:
    """§10.2 지표. `predicted` 에 gold 레코드가 하나라도 빠지면 거부한다.

    빠진 건을 오답으로 치든 빼고 재든, 어느 쪽도 "그 방식의 정확도" 가 아니다. 예측 파일을 잘못
    준 것이니 멈춘다. gold 밖 레코드의 예측(다른 split)은 세지 않는다.
    """
    missing = sorted(set(gold) - set(predicted))
    if missing:
        raise ValueError(f"예측이 없는 레코드 {len(missing)}건 {missing[:3]}")

    pairs = [
        (final["reason_label"], predicted[rid], final.get("evidence_grade"))
        for rid, final in gold.items()
    ]
    confusion = Counter((truth, guess) for truth, guess, _grade in pairs)
    correct = sum(count for (truth, guess), count in confusion.items() if truth == guess)

    per_class: dict[str, dict[str, float | int | None]] = {}
    for label in REASON_LABELS:
        support = sum(count for (truth, _), count in confusion.items() if truth == label)
        guessed = sum(count for (_, guess), count in confusion.items() if guess == label)
        hit = confusion[(label, label)]
        precision = hit / guessed if guessed else None
        recall = hit / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else (0.0 if support or guessed else None)
        )
        per_class[label] = {
            "support": support,
            "predicted": guessed,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    by_grade = {}
    for grade in EVIDENCE_GRADES:
        rows = [(truth, guess) for truth, guess, g in pairs if g == grade]
        hit = sum(truth == guess for truth, guess in rows)
        by_grade[grade] = {
            "n": len(rows),
            "correct": hit,
            "accuracy": hit / len(rows) if rows else None,
        }

    return {
        "n": len(pairs),
        "correct": correct,
        "accuracy": correct / len(pairs),
        "all_unk_accuracy": sum(truth == UNKNOWN_LABEL for truth, _, _ in pairs) / len(pairs),
        "per_class": per_class,
        "confusion": {
            truth: {
                guess: confusion[(truth, guess)]
                for guess in REASON_LABELS
                if confusion[(truth, guess)]
            }
            for truth in REASON_LABELS
        },
        "by_grade": by_grade,
    }


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_report(result: Mapping[str, Any], *, title: str) -> list[str]:
    """사람이 읽는 표. JSON 과 같은 숫자다."""
    lines = [
        f"## {title}",
        f"- 정확도 **{result['accuracy']:.1%}** ({result['correct']}/{result['n']})"
        f" - 모두 UNK 라고 답하면 {result['all_unk_accuracy']:.1%}",
        "",
        "| 이유 | 정답 수 | 예측 수 | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in result["per_class"].items():
        if not row["support"] and not row["predicted"]:
            continue
        lines.append(
            f"| {label} | {row['support']} | {row['predicted']} | {_ratio(row['precision'])}"
            f" | {_ratio(row['recall'])} | {_ratio(row['f1'])} |"
        )
    lines += [
        "",
        "근거 등급별 정확도 (정답 라벨의 등급): "
        + ", ".join(
            f"{grade} {row['correct']}/{row['n']}"
            for grade, row in result["by_grade"].items()
            if row["n"]
        ),
    ]
    lines += ["", "혼동 행렬 (행 = 정답, 열 = 예측, 0 은 비움)", ""]
    shown = [
        label
        for label in REASON_LABELS
        if result["per_class"][label]["support"] or result["per_class"][label]["predicted"]
    ]
    lines.append("| 정답 \\ 예측 | " + " | ".join(shown) + " |")
    lines.append("|---|" + "---:|" * len(shown))
    for truth in shown:
        row = result["confusion"][truth]
        lines.append(
            f"| {truth} | " + " | ".join(str(row.get(guess, "")) for guess in shown) + " |"
        )
    return lines


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자. 실행 방법은 모듈 독스트링 "실행" 절."""
    parser = argparse.ArgumentParser(
        prog="python -m classify.evaluate",
        description="예측 파일을 확정 라벨 한 split 과 비교해 정확도·F1·혼동 행렬 (#86, §10.2).",
    )
    parser.add_argument("predictions", type=Path, help="예측 JSONL (record_id, predicted_label)")
    parser.add_argument("--split", choices=SPLITS, default="val")
    parser.add_argument(
        "--labels", type=Path, default=DEFAULT_LABELS, help=f"기본 {DEFAULT_LABELS.as_posix()}"
    )
    parser.add_argument("--json", type=Path, default=None, help="결과를 JSON 으로도 저장")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """평가해서 표를 찍고, 원하면 JSON 으로 남긴다."""
    args = build_parser().parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        gold = load_gold(read_jsonl(args.labels), args.split)
        rows = read_jsonl(args.predictions)
        versions = sorted({str(row.get("version")) for row in rows})
        result = evaluate(gold, {str(row["record_id"]): row["predicted_label"] for row in rows})
    except (ValueError, KeyError) as error:
        print(f"평가할 수 없다: {error}", file=sys.stderr)
        return 2
    result = {
        "split": args.split,
        "predictions": args.predictions.as_posix(),
        "versions": versions,
        **result,
    }
    for line in format_report(
        result, title=f"{args.split} {result['n']}건 - {', '.join(versions)}"
    ):
        print(line)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nJSON: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
