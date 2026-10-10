"""이유 분류의 학습 부분 - 레코드 화면의 텍스트로 이유 8종의 확률을 낸다 (#84, #86). 담당: 희수

무엇을:
    레코드의 맥락 문장(`classify.rules.passages`), 함수·파일 이름, 같은 파일 추가 헝크, 그리고
    화면의 모양(PR·이슈·리뷰가 붙었나, 테스트인가, 추가 헝크·대체 코드가 있나)을 텍스트로 이어
    TF-IDF + 로지스틱 회귀로 이유 8종의 확률을 낸다. CHARTER §7 의 "규칙 + scikit-learn" 중
    scikit-learn 쪽이다. GPU 없이 노트북에서 도는 크기만 쓴다.

UNK 도 배운다 (m2, #86):
    m1 은 UNK 를 배우지 않고, 분류기가 "이유 키워드 문장도 대체 코드도 없으면 UNK" 로 정했다.
    500건 학습 300건 교차 검증에서 그 관문은 81건에 걸렸고 그중 사람 라벨 UNK 는 56건뿐이었다.
    반대로 사람 UNK 135건 중 79건은 관문을 통과했다 - 긴 PR 본문에는 거의 늘 `fix`·`remove` 같은
    키워드 문장이 있다. UNK 는 학습 데이터의 45%라 그것을 어떻게 가르느냐가 정확도의 절반이다.
    그래서 UNK 를 다른 이유와 같이 확률로 내고, 화면의 모양을 특징으로 준다 (사람이 UNK 를 고르는
    근거가 "맥락이 없다"·"추가 헝크가 없다" 같은 화면의 모양이다, 가이드 §6.3.1).

특징·하이퍼파라미터를 고른 방법:
    학습 300건 5겹 교차 검증 (`docs/reports/classifier_val.md`). 삭제된 코드 본문을 더하면 0.02
    낮아져 넣지 않았다. 클래스 가중치(balanced)는 UNK·DESIGN 이 많은 분포에서 소수 클래스를 과하게
    골라 0.04 낮아졌다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from classify.rules import passages

# 특징·모델을 바꾸면 올린다. 분류기 버전 문자열에 들어간다.
MODEL_VERSION = "m2"
# 로지스틱 회귀 규제 (sklearn `C`). 학습 300건 교차 검증에서 1.0 보다 10.0 이 0.02 높았다.
REGULARIZATION_C = 10.0
# 추가 헝크는 상위 10% 가 6천 자를 넘는다. 긴 헝크 하나가 특징을 다 차지하지 않게 자른다.
MAX_HUNK_TEXT_CHARS = 5000


def shape_tokens(record: dict[str, Any]) -> str:
    """화면의 모양을 낱말로. 맥락 텍스트에 나올 리 없는 `zz` 접두를 붙여 섞이지 않게 한다."""
    context = record.get("context") or {}
    flags = (
        ("pr", context.get("pr_number") is not None),
        ("issue", bool(context.get("issue_numbers"))),
        ("review", bool(context.get("review_comments"))),
        ("test", bool(record.get("is_test_code"))),
        ("hunks", bool(record.get("added_hunks_same_file"))),
        ("replacement", bool((record.get("replacement") or {}).get("code"))),
    )
    return " ".join(f"zz{'has' if present else 'no'}{name}" for name, present in flags)


def record_text(record: dict[str, Any]) -> str:
    """모델이 보는 텍스트. 맥락 문장 + 함수·파일 이름 + 화면 모양 + 같은 파일 추가 헝크.

    파일 이름은 규칙의 "이 함수를 가리키나" 판정에서는 뺐지만 (`rules.target_names`) 여기서는
    둔다. 어느 모듈에서 지워졌는지가 쓸모 있는 신호일 수 있고, 인용문이 아니라서 잘못 걸려도
    EXPLICIT 을 만들지 않는다.
    """
    lines = [passage.text for passage in passages(record)]
    lines.append(record.get("function_name") or "")
    lines.append(Path(record.get("file_path") or "").stem)
    lines.append(shape_tokens(record))
    hunks = record.get("added_hunks_same_file") or []
    added = " ".join(str(hunk.get("added_body") or "") for hunk in hunks if isinstance(hunk, dict))
    lines.append(added[:MAX_HUNK_TEXT_CHARS])
    return "\n".join(line for line in lines if line)


@dataclass
class ReasonModel:
    """TF-IDF + 로지스틱 회귀. 학습 전이거나 학습할 수 없으면 빈 확률을 낸다."""

    pipeline: Pipeline | None = None

    def fit(self, records: Sequence[dict[str, Any]], labels: Sequence[str]) -> ReasonModel:
        """학습한다. 이유가 두 종류 미만이면 학습하지 않는다.

        한 종류만으로는 로지스틱 회귀가 학습되지 않는다. 그때 예외를 내면 분류기 전체가
        멈추는데, 모델은 구성요소 중 하나라 빠져도 LLM 만으로 돈다. 빈 확률을 내고 계속한다.
        """
        texts = [record_text(record) for record in records]
        if len(set(labels)) < 2:
            self.pipeline = None
            return self
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("clf", LogisticRegression(C=REGULARIZATION_C, max_iter=3000)),
            ]
        )
        # m1 은 맥락이 모두 빈 레코드에서 TF-IDF 가 "empty vocabulary" 로 실패할 수 있어 그 예외를
        # 받았다. m2 는 `shape_tokens` 가 늘 낱말을 넣어 그 경우가 없다.
        pipeline.fit(texts, list(labels))
        self.pipeline = pipeline
        return self

    @property
    def trained(self) -> bool:
        """학습됐나. 안 됐으면 `predict_proba` 가 빈 확률을 낸다."""
        return self.pipeline is not None

    @property
    def classes(self) -> tuple[str, ...]:
        """학습에 나온 이유들. 학습 데이터에 없던 이유(500건 학습 split 의 PERF)는 확률이 0."""
        if self.pipeline is None:
            return ()
        return tuple(str(label) for label in self.pipeline.classes_)

    def predict_proba(self, record: dict[str, Any]) -> dict[str, float]:
        """이유별 확률. 학습 전이면 빈 dict - 분류기는 이 구성요소 없이 진행한다."""
        if self.pipeline is None:
            return {}
        probabilities = self.pipeline.predict_proba([record_text(record)])[0]
        return {label: float(p) for label, p in zip(self.classes, probabilities, strict=True)}
