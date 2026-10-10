"""분류 정확도 평가 테스트 (#86) - 지표 계산과 split 맞추기."""

import json

import pytest

from classify import evaluate as ev


def gold_rows():
    """학습 1건 + 검증 4건. 검증 정답: DEAD, DEAD, UNK, FEAT."""
    rows = [("t1", "train", "DESIGN", "INFERRED")]
    rows += [
        ("v1", "val", "DEAD", "EXPLICIT"),
        ("v2", "val", "DEAD", "INFERRED"),
        ("v3", "val", "UNK", "UNKNOWN"),
        ("v4", "val", "FEAT", "INFERRED"),
    ]
    return [
        {
            "record_id": rid,
            "split": split,
            "final": {"reason_label": label, "evidence_grade": grade},
        }
        for rid, split, label, grade in rows
    ]


def test_metrics_on_a_hand_checked_example():
    """정확도 3/4, DEAD P=1/1 R=1/2, FEAT 은 예측 2건 중 1건 - 손으로 센 값과 같아야 한다."""
    gold = ev.load_gold(gold_rows(), "val")
    predicted = {"v1": "DEAD", "v2": "FEAT", "v3": "UNK", "v4": "FEAT", "t1": "BUG"}
    result = ev.evaluate(gold, predicted)

    assert set(gold) == {"v1", "v2", "v3", "v4"}  # 다른 split 은 세지 않는다
    assert (result["n"], result["correct"], result["accuracy"]) == (4, 3, 0.75)
    assert result["all_unk_accuracy"] == 0.25
    dead, feat = result["per_class"]["DEAD"], result["per_class"]["FEAT"]
    assert (dead["precision"], dead["recall"]) == (1.0, 0.5)
    assert dead["f1"] == pytest.approx(2 / 3)
    assert (feat["support"], feat["predicted"]) == (1, 2)
    assert (feat["precision"], feat["recall"]) == (0.5, 1.0)
    assert result["per_class"]["BUG"]["f1"] is None  # 정답도 예측도 없는 이유
    assert result["confusion"]["DEAD"] == {"DEAD": 1, "FEAT": 1}
    assert result["by_grade"]["INFERRED"] == {"n": 2, "correct": 1, "accuracy": 0.5}


def test_a_missing_prediction_is_refused_not_counted_wrong():
    """빠진 건을 오답이나 제외로 치면 그 방식의 정확도가 아니다."""
    gold = ev.load_gold(gold_rows(), "val")

    with pytest.raises(ValueError, match="예측이 없는 레코드 1건"):
        ev.evaluate(gold, {"v1": "DEAD", "v2": "DEAD", "v3": "UNK"})


def test_an_unsettled_final_is_refused():
    rows = gold_rows()
    rows[1]["final"]["reason_label"] = None

    with pytest.raises(ValueError, match="확정되지 않았다"):
        ev.load_gold(rows, "val")


def test_cli_prints_the_table_and_writes_json(tmp_path, capsys):
    labels, predictions, out = tmp_path / "l.jsonl", tmp_path / "p.jsonl", tmp_path / "r.json"
    labels.write_text("\n".join(json.dumps(row) for row in gold_rows()), encoding="utf-8")
    preds = [("v1", "DEAD"), ("v2", "DEAD"), ("v3", "DESIGN"), ("v4", "FEAT")]
    predictions.write_text(
        "\n".join(
            json.dumps({"record_id": rid, "predicted_label": label, "version": "x1"})
            for rid, label in preds
        ),
        encoding="utf-8",
    )

    code = ev.main([str(predictions), "--labels", str(labels), "--json", str(out)])

    assert code == 0
    assert "정확도 **75.0%** (3/4)" in capsys.readouterr().out
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert (saved["split"], saved["versions"], saved["correct"]) == ("val", ["x1"], 3)
