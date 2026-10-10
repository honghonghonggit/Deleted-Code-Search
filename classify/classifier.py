"""우리 방식 - 맥락 결합 + 규칙 + scikit-learn + LLM 으로 이유와 근거 등급 (#84, #86). 담당: 희수

무엇을:
    §4.4 레코드 하나를 받아 `reason`(이유 8종 + 근거 등급 + 근거 문장 + 신뢰도)을 낸다.
    CHARTER §4.2 ③ "우리 방식" 이고, 게이트 2(§10.2)에서 기준선 A/B 를 이겨야 하는 쪽이다.

어떻게 고르나 (#86):
    1. 이유 8종(UNK 포함)마다 점수 = 모델 확률(`classify.model`) + LLM 이 고른 라벨에 `WEIGHT_LLM`
    2. 가장 높은 라벨. 동점이면 이유 문장이 받치는 라벨, 그다음 가이드 §11-1 우선순위
    3. UNK 면 UNKNOWN. 아니면 고른 라벨을 가장 직접 받치는 근거로 등급을 정한다 (`_grade`)

    규칙(`classify.rules`)은 라벨을 고르지 않고 근거 등급과 인용에만 쓴다. #84 에서는 규칙 문장의
    키워드 라벨이 표를 던졌는데, 학습 300건 5겹 교차 검증에서 그 표가 BUG 를 98건(정답 6건) 내며
    정확도를 0.46 → 0.33 으로 끌어내렸다 - 긴 PR 본문에는 `fix` 같은 낱말이 거의 늘 있다. 같은
    이유로 "키워드 문장도 대체 코드도 없으면 UNK" 관문도 뺐다 (`classify.model` 독스트링).

LLM 은 후보와 근거 문장만 (ADR-005):
    LLM 답은 라벨 점수에 `WEIGHT_LLM` 을 더할 뿐이고, 학습 데이터로 배운 모델 확률과 겨룬다.
    근거 문장을 써 줄 수는 있다 - ADR-005 가 허락한 "근거 작성" 이다. EXPLICIT 의 근거 문장은
    LLM 이 쓰지 않는다. 원문 인용이어야 한다 (가이드 §6.1).

고른 방법:
    `WEIGHT_LLM`, 모델 특징·규제(`classify.model`), LLM 프롬프트(c3)는 학습 300건 5겹 교차 검증으로
    고르고 검증 100건으로 한 번 확인했다. 시도한 것 전부와 버린 것은
    `docs/reports/classifier_val.md`.
    근거 ②~⑥(가이드 §6.2.1)의 규칙은 아직 없다 - 그 판단은 LLM 이 화면(추가 헝크·삭제 본문)을
    보고 한다. "이유 문장이 이 함수를 가리키지 않을 때" 의 신뢰도는 #84 값 그대로다.

실행:
    python -m classify.classifier --records records.jsonl --labels merged.jsonl --out out.jsonl
    python -m classify.classifier ... --llm        # LLM 후보도 쓴다 (NVIDIA_API_KEY, #59)
    python -m classify.classifier ... --llm --split val   # 학습 split 으로 학습, 검증 split 예측
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from classify.baseline_keyword import KEYWORD_RULES
from classify.baseline_llm import (
    CALL_ERRORS,
    DEFAULT_PROVIDER,
    MAX_DIFF_CHARS,
    PROVIDERS,
    LlmBaseline,
    caller_from_env,
    parse_answer,
)
from classify.baselines import UNKNOWN_LABEL, record_id_of
from classify.labels import EVIDENCE_GRADES, REASON_LABELS
from classify.model import MODEL_VERSION, ReasonModel
from classify.rules import RULES_VERSION, ReasonSentence, find_reason_sentences, passages
from pipeline.select_repos import load_env_file

METHOD_OURS = "ours"
CANDIDATE_PROMPT_VERSION = "c3"
CLASSIFIER_VERSION = f"{RULES_VERSION}+{MODEL_VERSION}+{CANDIDATE_PROMPT_VERSION}"

EXPLICIT, INFERRED, UNKNOWN = EVIDENCE_GRADES
# 점수가 같을 때의 순서. 기준선 A 규칙 순서 = 가이드 §11-1 우선순위 (SEC > LIB > ... > DESIGN).
PRIORITY: tuple[str, ...] = tuple(label for label, _patterns in KEYWORD_RULES)

# LLM 이 고른 라벨에 더하는 점수. 모델 확률(합 1)과 겨룬다. 학습 300건 5겹 교차 검증에서
# 0.4~0.7 이 0.65~0.67 로 평평했고 그 가운데를 골랐다 (0 이면 모델만 0.60, 1 이상이면 LLM 답
# 그대로 0.62). `docs/reports/classifier_val.md`.
WEIGHT_LLM = 0.5

# EXPLICIT 은 1.0 고정 (가이드 §6.1, §11-3).
EXPLICIT_CONFIDENCE = 1.0
# 이유 문장은 있는데 이 함수를 이름으로 가리키지 않는다 - 가이드 §6.1.1 E2 를 못 넘어 INFERRED.
# 0.5~0.8 구간(§6.2.2)의 가운데. 개발 102건의 사람 라벨도 0.55~0.6 에 몰려 있다.
UNREACHED_SENTENCE_CONFIDENCE = 0.6
# INFERRED 는 1.0 을 쓰지 않는다 (§6.2.2). 대체 코드 근거는 이 값과 `replacement.confidence`
# 중 작은 쪽 - 파이프라인이 0.7 만 확신하는 대체 코드로 0.8+ 를 줄 수 없다.
INFERRED_CONFIDENCE_CAP = 0.9
# 이 아래는 INFERRED 가 아니라 UNKNOWN 이다 (§6.2.2).
INFERRED_FLOOR = 0.5

MAX_CONTEXT_CHARS = 6000
MAX_REPLACEMENT_CHARS = 2000
MAX_HUNKS_CHARS = 3000
# 기준선 B 캐시 옆. 커밋하는 이유는 `classify.baseline_llm.DEFAULT_CACHE_DIR`.
DEFAULT_CACHE_DIR = Path("datasets") / "llm_cache" / "classifier"
# `--split` 으로 예측할 split -> 모델을 학습할 split. 검증은 학습 300건으로, 최종 test 는 모델·
# 가중치를 고정한 뒤 학습+검증 400건으로 다시 학습해 한 번 돈다 (`docs/evaluation.md` #86 절).
TRAIN_SPLITS = {"val": ("train",), "test": ("train", "val")}

# c2 프롬프트에 넣는 라벨 가이드 v3 요약 (§3 판정 순서, §4, §5, §6.2.1, §6.3.3, §6.4). 사람 라벨이
# 이 기준으로 정해졌으니 LLM 도 같은 기준으로 답해야 한다. 이유 정의와 헷갈리는 쌍은 라벨러용
# AI 보조 프롬프트(`docs/labeling_ai_prompt.md`)의 문구를 줄여 썼다 - 같은 기준을 두 벌로 쓰지
# 않으려고. 체크리스트 전체를 넣지 않은 것은 답이 길어져 `MAX_ANSWER_TOKENS`(256, 게이트 2 사전
# 등록)에 안 들어가서다. 기준선 B 는 이것을 받지 않는다 - 기준선 B 의 정의가 "직접 질의" 다.
CANDIDATE_GUIDE = """이유 8종:
- BUG: 잘못된 동작(오류·크래시·경계 조건·경쟁 조건)을 고치려고 지웠다. 'fix' 낱말만으로는 아니다
- PERF: 결과는 같고 시간·메모리 비용을 줄이려고 지웠다. 테스트·벤치마크 삭제에는 쓰지 않는다
- SEC: 취약점·위험 패턴(신뢰할 수 없는 입력의 eval·pickle·shell, 약한 암호, 비밀 노출)을 없앴다
- LIB: 이 커밋에서 직접 구현을 외부·표준 라이브러리 호출로 바꿨다
- DEAD: 호출되지 않아 지웠다 (테스트면 검증할 대상이 없어져서)
- DESIGN: 구조·책임·인터페이스를 바꾸면서 이 코드가 맞지 않게 됐다. 같은 일은 다른 자리에서 계속된다
- FEAT: 사용자가 쓰던 기능 자체를 없앴다 (폐기·지원 종료·옵션 제거)
- UNK: 아래 화면으로는 근거를 댈 수 없다

판단 순서:
1. 배울 게 없는 삭제면 UNK
   - 같은 본문이 추가된 코드에 다른 이름·위치로 있다 (순수 이동·이름 바꾸기)
   - 추가된 코드에 같은 줄이 포맷·문법만 바뀌어 있다
   - 경로에 vendor/·vendored/·third_party/ 가 있거나 생성 코드다
   - 커밋 메시지·PR 이 코드를 다른 저장소로 옮긴다(move·migrate·extract·split out)고 그 저장소를 \
밝히고, 이 파일이 옮기는 대상 안에 있다
2. 맥락 문장이 이 함수·파일·모듈을 이름으로 가리키며 삭제의 "왜"를 말하면 그 이유를 따른다. \
지웠다·옮겼다·바꿨다·다시 썼다·되돌렸다(revert)는 사실이나 fix·cleanup·refactor 같은 낱말만 \
있으면 이유가 아니다
3. 그런 문장이 없으면 화면의 코드로만 추론한다 - 대체 코드나 추가된 코드에서 삭제된 일을 이어받는 \
줄을 가리킬 수 있나, 추가된 코드에서 이 함수를 부르던 자리가 바뀌었나, 삭제된 본문 자체가 이유를 \
드러내나 (deprecated 경고, 빈 스텁, 제거 예정 주석, @skip)
4. 커밋이 큰 작업(전환·재작성·재구성·이동·되돌림)을 하고 이 함수가 그 영역에 있다는 것만으로는 \
근거가 아니다. 네 근거 문장에서 함수 이름만 바꿔도 같은 커밋의 다른 삭제에 그대로 맞으면 근거가 \
아니다. 근거가 없으면 UNK
5. 이유가 둘이면 수단이 아니라 이유, 결과가 아니라 원인을 고른다 ("refactor to fix race" 는 BUG)

테스트 함수 (테스트 코드: 예):
- 이 테스트가 지워졌다는 사실 자체는 근거가 아니다. 테스트 대상의 사정(폐기·재편)으로 FEAT·DESIGN \
을 고르지 않는다
- 이유 문장이 이 테스트를 이름으로 가리키면 그 이유를 따른다
- 테스트가 부르거나 import 하는 대상을 이 커밋에서 지웠다(remove·delete·drop)고 문장이 그 이름으로 \
말하면 DEAD
- 그 밖에는 삭제된 본문 자체(@skip·@xfail, 단정문 없음)가 이유를 드러낼 때만 그 이유, 아니면 UNK

헷갈리는 쌍:
- LIB vs DESIGN: 외부·표준 라이브러리가 일을 이어받으면 LIB, \
우리 코드의 다른 자리가 이어받으면 DESIGN
- LIB vs DEAD: 교체가 이 커밋에서 일어났으면 LIB, 이미 끝났고 잔재만 치우면 DEAD
- DEAD vs FEAT: 공개 표면(공개 API·CLI·설정·문서)에서 사라지면 FEAT, \
내부 함수가 호출자를 잃었으면 DEAD
- DESIGN vs FEAT: 같은 일을 다른 API 로 계속 할 수 있으면 DESIGN, 못 하게 됐으면 FEAT
- DEAD vs DESIGN: 호출자가 없어서면 DEAD, 호출 경로를 옮겨서면 DESIGN"""


@dataclass(frozen=True)
class Classification:
    """레코드 하나에 대한 우리 방식의 답."""

    record_id: str
    label: str
    evidence_grade: str
    evidence_text: str
    evidence_source: str
    evidence_locator: str
    confidence: float
    version: str = CLASSIFIER_VERSION
    scores: dict[str, float] = field(default_factory=dict)
    """허락된 라벨별 합산 점수. 왜 그 라벨인지 사람이 볼 수 있어야 가중치를 고칠 수 있다."""
    note: str = ""

    def __post_init__(self) -> None:
        """라벨·등급이 정의된 값인지 확인한다. 밖의 값이 예측 파일에 섞이면 평가가 조용히 틀린다."""
        if self.label not in REASON_LABELS:
            raise ValueError(f"{self.label!r} 은 §4.2 ③ 8종이 아니다")
        if self.evidence_grade not in EVIDENCE_GRADES:
            raise ValueError(f"{self.evidence_grade!r} 은 근거 3등급이 아니다")

    def to_schema_reason(self) -> dict[str, Any]:
        """§4.4 `reason` 그대로. 필드를 늘리거나 이름을 바꾸지 않는다 (§13).

        `evidence_source`·`evidence_locator` 는 §4.4 `reason` 에 칸이 없어 여기 넣지 않는다.
        예측 파일(`to_dict`)에만 남는다 - 평가 때 근거 위치를 대조하는 데 쓴다.
        """
        return {
            "label": self.label,
            "evidence_grade": self.evidence_grade,
            "evidence_text": self.evidence_text or None,
            "confidence": self.confidence,
            "classifier_version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        """예측 파일 한 줄. 기준선 예측(`classify.baselines.Prediction`)과 같은 키로 시작한다.

        `predicted_label` 인 이유도 기준선과 같다 - 사람 라벨(`reason_label`)과 이름을 갈라
        섞여 들어가면 바로 보이게 한다 (ADR-005).
        """
        return {
            "record_id": self.record_id,
            "predicted_label": self.label,
            "method": METHOD_OURS,
            "version": self.version,
            "evidence_grade": self.evidence_grade,
            "evidence_text": self.evidence_text,
            "evidence_source": self.evidence_source,
            "evidence_locator": self.evidence_locator,
            "confidence": self.confidence,
            "scores": {label: round(score, 4) for label, score in self.scores.items()},
            "note": self.note,
        }


# --------------------------------------------------------------------------------------
# LLM 후보 (ADR-005 - 후보 생성·근거 작성만)
# --------------------------------------------------------------------------------------


def build_candidate_prompt(record: dict[str, Any]) -> str:
    """LLM 에 줄 본문. 기준선 B 와 달리 **라벨러가 보는 화면 전체**를 준다 - 그 차이가 우리
    방식이다.

    맥락은 `classify.rules.passages` 가 만든 문장을 위치와 함께 준다. 같은 문장을 규칙과 LLM
    이 함께 보므로, LLM 이 근거로 든 문장이 어디 있었는지 대조할 수 있다.

    c2 (#86) 가 c1 에 더한 것: 파일·함수·테스트 여부, 같은 파일 추가 헝크, 그리고 라벨 가이드 v3
    의 판정 순서와 헷갈리는 쌍(`CANDIDATE_GUIDE`). c1 은 이유 정의 한 줄씩과 맥락·삭제 코드·대체
    코드만 줬다. 사람 라벨은 추가 헝크를 보고 가이드 순서대로 정한 것이라, 그것을 안 주고 같은
    답을 기대할 수 없다 (val 비교는 `docs/reports/classifier_val.md`).
    """
    context_lines: list[str] = []
    used = 0
    for passage in passages(record):
        line = f"[{passage.locator}] {passage.text}"
        if used + len(line) > MAX_CONTEXT_CHARS:
            context_lines.append("... (이하 생략)")
            break
        context_lines.append(line)
        used += len(line)
    context = "\n".join(context_lines) or "(맥락 없음)"
    deleted = (record.get("deleted_body") or "(삭제 코드 없음)")[:MAX_DIFF_CHARS]
    replacement = ((record.get("replacement") or {}).get("code") or "(없음)")[
        :MAX_REPLACEMENT_CHARS
    ]
    function = record.get("function_signature") or record.get("function_name") or ""
    return (
        "아래 함수가 왜 삭제됐는지 이유 8종 중 하나로 분류하라.\n\n"
        f"{CANDIDATE_GUIDE}\n\n"
        f"파일: {record.get('file_path') or ''}\n"
        f"함수: {function}\n"
        f"테스트 코드: {'예' if record.get('is_test_code') else '아니오'}\n\n"
        f"맥락 (줄 앞 [ ] 는 출처):\n{context}\n\n"
        f"삭제된 코드:\n```\n{deleted}\n```\n\n"
        f"같은 커밋이 이 파일에 추가한 코드:\n```\n{added_hunks_text(record)}\n```\n\n"
        f"같은 자리에 들어온 대체 코드:\n```\n{replacement}\n```\n\n"
        "답은 정확히 한 줄로, `라벨|근거` 형식으로만 쓴다. 라벨은 위 8개 중 하나이고, 근거는 "
        "위 화면의 어디를 보고 판단했는지 한 문장으로 쓴다."
    )


def added_hunks_text(record: dict[str, Any]) -> str:
    """같은 파일 추가 헝크를 `[k/N] new_start S` 머리 줄과 본문으로 (`tools/label_cli` 화면과
    같은 꼴).

    `MAX_HUNKS_CHARS` 에서 자른다. 추가 헝크는 중앙값이 0 이지만 상위 10% 는 6천 자를 넘는다.
    """
    hunks = [hunk for hunk in record.get("added_hunks_same_file") or [] if isinstance(hunk, dict)]
    if not hunks:
        return "(이 커밋이 이 파일에 추가한 줄 없음)"
    pieces = [
        f"[{position}/{len(hunks)}] new_start {hunk.get('new_start')}\n"
        f"{hunk.get('added_body') or ''}"
        for position, hunk in enumerate(hunks, start=1)
    ]
    text = "\n".join(pieces)
    return text if len(text) <= MAX_HUNKS_CHARS else text[:MAX_HUNKS_CHARS] + "\n... (이하 생략)"


@dataclass
class LlmCandidate:
    """LLM 에게 라벨 후보와 근거 문장을 묻는다. 캐시·호출기는 기준선 B 것을 그대로 쓴다."""

    runner: LlmBaseline
    failures: dict[str, int] = field(default_factory=dict)

    def propose(self, record: dict[str, Any]) -> tuple[str, str] | None:
        """(라벨, 근거 문장). 호출이 실패하거나 답을 못 읽으면 `None` - 분류는 LLM 없이 계속한다.

        LLM 은 세 구성요소 중 하나다. 한 건의 호출 실패로 배치 전체가 멈추면 안 되고, 실패를
        UNK 로 바꿔 한 표를 주면 UNK 쪽으로 기운다. 표를 주지 않는 것이 맞다.
        """
        try:
            text = self.runner.ask(build_candidate_prompt(record))
        except CALL_ERRORS as error:
            self._fail(f"호출 실패: {type(error).__name__}")
            return None
        label, reason, problem = parse_answer(text)
        if problem:
            self._fail(problem.split(":")[0])
            return None
        return label, reason

    def _fail(self, reason: str) -> None:
        """실패 사유별 건수를 센다. CLI 가 끝에 보고한다."""
        self.failures[reason] = self.failures.get(reason, 0) + 1


# --------------------------------------------------------------------------------------
# 분류
# --------------------------------------------------------------------------------------


@dataclass
class Classifier:
    """모델 + (선택) LLM 이 라벨을 고르고, 규칙이 근거 등급과 인용을 댄다."""

    model: ReasonModel = field(default_factory=ReasonModel)
    llm: LlmCandidate | None = None

    @property
    def version(self) -> str:
        """분류기 버전. LLM 을 쓰면 모델 이름까지 넣는다.

        LLM 유무·모델이 다른 두 예측 파일이 같은 버전 문자열을 달면 나중에 가를 수 없다.
        기준선 B 가 `프롬프트/모델` 을 버전에 넣는 것과 같은 이유다.
        """
        if self.llm is None:
            return CLASSIFIER_VERSION
        return f"{CLASSIFIER_VERSION}+llm:{self.llm.runner.model}"

    def classify(self, record: dict[str, Any]) -> Classification:
        """레코드 하나. 순서는 모듈 독스트링 "어떻게 고르나"."""
        return replace(self._classify(record), version=self.version)

    def _classify(self, record: dict[str, Any]) -> Classification:
        """`classify` 의 본체. 버전은 `classify` 가 덧씌운다."""
        record_id = record_id_of(record)
        sentences = find_reason_sentences(record)
        candidate = self.llm.propose(record) if self.llm is not None else None
        scores = self._scores(record, candidate)
        # 모델이 학습되지 않았고 LLM 도 답하지 않았으면 어느 이유인지 말할 신호가 없다. 그대로
        # 두면 동점 규칙이 §11-1 첫 순위(SEC)를 고른다 - `--records` 에 라벨과 이어지는 건이
        # 없을 때 실제로 그렇게 된다.
        if max(scores.values()) <= 0.0:
            return Classification(
                record_id, UNKNOWN_LABEL, UNKNOWN, "", "", "", 0.0, scores=scores, note="신호 없음"
            )
        backed = {sentence.label for sentence in sentences}
        label = max(
            scores, key=lambda reason: (scores[reason], reason in backed, -_priority(reason))
        )
        if label == UNKNOWN_LABEL:
            return Classification(record_id, UNKNOWN_LABEL, UNKNOWN, "", "", "", 0.0, scores=scores)
        replacement = _usable_replacement(record)
        return self._grade(record_id, label, sentences, replacement, candidate, scores)

    def _scores(
        self, record: dict[str, Any], candidate: tuple[str, str] | None
    ) -> dict[str, float]:
        """이유 8종(UNK 포함)의 점수 = 모델 확률 + LLM 이 고른 라벨에 `WEIGHT_LLM`.

        규칙 문장의 키워드 라벨은 표를 던지지 않는다 (모듈 독스트링 "어떻게 고르나").
        """
        probabilities = self.model.predict_proba(record)
        llm_label = candidate[0] if candidate else None
        return {
            reason: probabilities.get(reason, 0.0) + (WEIGHT_LLM if reason == llm_label else 0.0)
            for reason in REASON_LABELS
        }

    def _grade(
        self,
        record_id: str,
        label: str,
        sentences: Sequence[ReasonSentence],
        replacement: tuple[str, float] | None,
        candidate: tuple[str, str] | None,
        scores: dict[str, float],
    ) -> Classification:
        """고른 라벨을 **가장 직접 받치는** 근거로 등급을 정한다. 아래 순서대로 처음 맞는 것.

        1. 그 라벨의 이유 문장이 삭제된 함수를 이름으로 가리킨다 -> EXPLICIT (원문 인용)
        2. 그 라벨의 이유 문장은 있는데 이 함수까지 닿지 않는다 -> INFERRED (가이드 §6.1.1 E2)
        3. 대체 코드가 있다 -> INFERRED, 근거 ① (`diff:replacement`)
        4. LLM 이 같은 라벨을 골랐다 -> INFERRED, 하한 0.5, 근거 문장은 LLM 이 쓴 것
        5. 이유 문장은 있는데 키워드가 다른 이유를 가리켰다 -> INFERRED, 하한 0.5, 그 문장
        6. 모델만 골랐다 -> INFERRED, 하한 0.5, 근거 문장 없음. 4~6 은 가장 약한 경우라 신뢰도를
           올리지 않는다

        2 가 3 보다 먼저인 것은 신뢰도 순서가 아니다 (2 는 0.6, 3 은 최대 0.9). 같은 라벨의 이유
        문장은 **왜**를 말하고, 대체 코드는 무엇이 이어받았는지만 말해 어느 이유도 받치지 않는다.
        고른 라벨의 근거로는 앞이 더 직접적이다.

        EXPLICIT 은 **같은 라벨**의 문장으로만 준다 - 인용문이 그 이유를 말해야 한다 (§6.1).
        """
        own = [sentence for sentence in sentences if sentence.label == label]
        named = [sentence for sentence in own if sentence.names_target]
        if named:
            passage = named[0].passage
            return Classification(
                record_id,
                label,
                EXPLICIT,
                passage.text,
                passage.source,
                passage.locator,
                EXPLICIT_CONFIDENCE,
                scores=scores,
            )
        if own:
            passage = own[0].passage
            return Classification(
                record_id,
                label,
                INFERRED,
                passage.text,
                passage.source,
                passage.locator,
                UNREACHED_SENTENCE_CONFIDENCE,
                scores=scores,
                note="이유 문장이 삭제된 함수를 이름으로 가리키지 않는다 (가이드 §6.1.1 E2)",
            )
        if replacement is not None:
            code, confidence = replacement
            if candidate is not None and candidate[0] == label and candidate[1]:
                text, note = candidate[1], "근거 문장: LLM 작성 (ADR-005)"
            else:
                text, note = f"같은 자리에 대체 코드가 들어왔다: {_first_line(code)}", ""
            return Classification(
                record_id,
                label,
                INFERRED,
                text,
                "diff",
                "diff:replacement",
                confidence,
                scores=scores,
                note=note,
            )
        if candidate is not None and candidate[0] == label and candidate[1]:
            # LLM 이 화면(추가 헝크·삭제 본문·맥락)을 보고 댄 근거다. 위치를 규칙으로 확인하지
            # 못했으니 source·locator 는 비우고 하한 신뢰도를 쓴다.
            return Classification(
                record_id,
                label,
                INFERRED,
                candidate[1],
                "",
                "",
                INFERRED_FLOOR,
                scores=scores,
                note="근거 문장: LLM 작성 (ADR-005)",
            )
        if sentences:
            passage = sentences[0].passage
            return Classification(
                record_id,
                label,
                INFERRED,
                passage.text,
                passage.source,
                passage.locator,
                INFERRED_FLOOR,
                scores=scores,
                note=f"문장 키워드는 {sentences[0].label} 을 가리켰고 라벨은 모델·LLM 이 골랐다",
            )
        return Classification(
            record_id,
            label,
            INFERRED,
            "",
            "",
            "",
            INFERRED_FLOOR,
            scores=scores,
            note="모델만 이 라벨을 골랐다 - 화면에서 짚은 근거가 없다",
        )


def _usable_replacement(record: dict[str, Any]) -> tuple[str, float] | None:
    """근거 ① 로 쓸 수 있는 대체 코드와 그 신뢰도. 못 쓰면 `None`.

    `code` 가 `null` 이거나 공백뿐이면 근거 ① 을 쓸 수 없다 (가이드 §6.2.3). 신뢰도가 0.5
    아래로 떨어지면 INFERRED 가 아니라 UNKNOWN 이다 (§6.2.2).

    신뢰도는 유한한 숫자만 믿는다. `json.loads` 는 `NaN` 을 받는데 `min(0.9, nan)` 은 0.9 이고,
    `True` 는 int 라 1.0 이 된다 - 둘 다 근거 없는 0.9 로 새어 나간다.
    """
    replacement = record.get("replacement") or {}
    code = replacement.get("code")
    if not isinstance(code, str) or not code.strip():
        return None
    raw = replacement.get("confidence")
    usable = isinstance(raw, int | float) and not isinstance(raw, bool) and math.isfinite(raw)
    confidence = min(INFERRED_CONFIDENCE_CAP, float(raw)) if usable else 0.0
    return (code, confidence) if confidence >= INFERRED_FLOOR else None


def _priority(label: str) -> int:
    """가이드 §11-1 우선순위에서의 자리. 작을수록 먼저다."""
    return PRIORITY.index(label) if label in PRIORITY else len(PRIORITY)


def _first_line(code: str) -> str:
    """코드의 첫 비어 있지 않은 줄 - 대체 코드를 한 줄로 가리킬 때 쓴다."""
    return next((line.strip() for line in code.splitlines() if line.strip()), "")


# --------------------------------------------------------------------------------------
# 입력·학습
# --------------------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL 을 읽는다. 빈 줄은 건너뛴다. `splitlines()` 를 쓰지 않는 이유는 `labels.read_jsonl`.

    그 함수와 달리 파일이 없으면 오류다 - 라벨 파일이 없는데 빈 목록으로 읽으면 모델이 조용히
    학습되지 않는다.
    """
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_final_labels(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    """병합 라벨 파일(`eval/gate1_merge.py` 출력) -> `{record_id: final.reason_label}`.

    확정된 것만 쓴다. 두 사람이 갈려 확정되지 않은 건(`final.reason_label` 이 `null`)을 한쪽
    라벨로 채우면 그 사람의 판단을 정답으로 학습하는 셈이다.
    """
    labels: dict[str, str] = {}
    for row in rows:
        label = (row.get("final") or {}).get("reason_label")
        record_id = str(row.get("record_id") or "").strip()
        if label in REASON_LABELS and record_id:
            labels[record_id] = label
    return labels


def train_model(records: Sequence[dict[str, Any]], labels: dict[str, str]) -> ReasonModel:
    """레코드와 확정 라벨을 `id` 로 이어 모델을 학습한다."""
    pairs = [(record, labels[key]) for record in records if (key := record_id_of(record)) in labels]
    return ReasonModel().fit([record for record, _ in pairs], [label for _, label in pairs])


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자. 실행 방법은 모듈 독스트링 "실행" 절."""
    parser = argparse.ArgumentParser(
        prog="python -m classify.classifier",
        description="우리 방식 - 규칙 + scikit-learn + LLM 으로 이유와 근거 등급 (#84).",
    )
    parser.add_argument("--records", type=Path, required=True, help="§4.4 레코드 JSONL")
    parser.add_argument(
        "--labels", type=Path, required=True, help="확정 라벨 (병합 파일, 모델 학습용)"
    )
    parser.add_argument("--out", type=Path, default=None, help="예측 JSONL 저장 경로")
    parser.add_argument("--llm", action="store_true", help="LLM 후보도 쓴다 (API 호출, #59)")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=None, help="LLM 모델 (기본: 공급자별 고정 모델)")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--split",
        choices=sorted(TRAIN_SPLITS),
        default=None,
        help="라벨 파일의 split 으로 나눠, 학습 split 으로 학습하고 이 split 만 예측한다 (#86)",
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="--split test 를 허락한다. test 는 최종 1회만 (게이트 2 사전 등록)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """분류하고 분포를 보고한다. **정확도는 내지 않는다** - 정확도는 #86 이 test 로 잰다."""
    args = build_parser().parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    records = load_jsonl(args.records)
    if not records:
        print("레코드가 없다.", file=sys.stderr)
        return 1
    label_rows = load_jsonl(args.labels)
    labels = load_final_labels(label_rows)
    training = records
    if args.split:
        if args.split == "test" and not args.final_test:
            print(
                "test 는 최종 1회만 돈다 (게이트 2 사전 등록). "
                "H 를 기록한 뒤 --final-test 로 돌려라.",
                file=sys.stderr,
            )
            return 2
        split_of = {str(row.get("record_id")): row.get("split") for row in label_rows}
        training = [r for r in records if split_of.get(record_id_of(r)) in TRAIN_SPLITS[args.split]]
        records = [r for r in records if split_of.get(record_id_of(r)) == args.split]
        if not training or not records:
            print(
                f"split 으로 나눈 학습 {len(training)}건 / 예측 {len(records)}건 - 비었다.",
                file=sys.stderr,
            )
            return 2
    model = train_model(training, labels)

    llm = None
    if args.llm:
        load_env_file(args.env_file)
        caller = caller_from_env(args.provider)
        if caller is None:
            return 2
        cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
        runner = LlmBaseline(
            caller,
            model=args.model or PROVIDERS[args.provider].model,
            cache_dir=cache_dir,
            prompt_version=CANDIDATE_PROMPT_VERSION,
        )
        llm = LlmCandidate(runner)

    classifier = Classifier(model=model, llm=llm)
    results = [classifier.classify(record) for record in records]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        print(f"예측: {args.out} ({len(results)}건)", file=sys.stderr)

    print(f"우리 방식 ({classifier.version}) - {len(results)}건")
    joined = sum(1 for record in training if record_id_of(record) in labels)
    print(
        f"  모델 학습: {'예' if model.trained else '아니오'}"
        f" (확정 라벨 {len(labels)}건 중 학습 레코드와 이어진 {joined}건)"
    )
    if args.split:
        trained_on = "+".join(TRAIN_SPLITS[args.split])
        print(f"  {trained_on} 로 학습, {args.split} 예측 - 정확도는 classify.evaluate 로 잰다.")
    else:
        print("  학습에 쓴 레코드도 예측에 들어간다. 이 분포로 정확도를 말하지 않는다 (#86).")
    for name, counter in (
        ("라벨", Counter(result.label for result in results)),
        ("등급", Counter(result.evidence_grade for result in results)),
    ):
        detail = ", ".join(f"{key} {count}" for key, count in counter.most_common())
        print(f"  {name}: {detail}")
    if llm is not None:
        print(f"  LLM: API {llm.runner.calls}회 / 캐시 {llm.runner.cache_hits}회")
        if llm.failures:
            print(f"  LLM 실패: {', '.join(f'{k} {v}건' for k, v in llm.failures.items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
