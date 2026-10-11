"""기준선 B: LLM 직접 질의 (§10.2). 담당: 희수 (hs)

무엇을:
    diff 와 커밋 메시지를 LLM 에 주고 §4.2 ③ 8종 중 하나를 고르게 한다. 기준선 A 보다 강한
    상대이고, 게이트 2(§9 6주차)에서 우리 방식이 이겨야 하는 쪽이다.

이 출력은 정답이 아니다 (ADR-005):
    LLM 출력을 라벨 자리에 쓰면 §10.2 평가가 자기 참조로 무너진다. 우리 방식과 기준선 B 를
    같은 LLM 답으로 채점하는 꼴이 되기 때문이다. 그래서 출력 필드는 `predicted_label` 이고
    (사람 라벨은 `reason_label`), 이 파일은 `datasets/labels/` 아래에 아무것도 쓰지 않는다.

캐시하는 이유:
    같은 레코드를 다시 질의하면 돈이 들고, 모델이 매번 다른 답을 줄 수 있어 재현이 안 된다.
    프롬프트·모델·입력이 같으면 캐시를 쓴다. 캐시 키에 프롬프트 버전이 들어가므로 프롬프트를
    고치면 자동으로 다시 묻는다.

실행:
    python -m classify.baseline_llm --input records.jsonl --out predictions.jsonl
    python -m classify.baseline_llm --input records.jsonl --dry-run   # 프롬프트만 확인
    python -m classify.baseline_llm ... --provider anthropic            # 유료로 바꿀 때

    키(NVIDIA_API_KEY, ANTHROPIC_API_KEY)는 .env 에서만 읽는다 (§8.4).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from classify.baselines import (
    METHOD_LLM,
    UNKNOWN_LABEL,
    Prediction,
    commit_message_of,
    record_id_of,
    summarize,
    write_predictions,
)
from classify.labels import REASON_LABELS, read_jsonl
from pipeline.select_repos import load_env_file

# 프롬프트를 고치면 올린다. 캐시 키와 예측 파일에 함께 들어가므로 어느 프롬프트로 낸 답인지 남는다.
PROMPT_VERSION = "b1"

# 공급자 (#59). NVIDIA API 무료 한도로 먼저 가고, 한도나 응답 품질로 안 되면 Anthropic 유료로
# 바꾼다 (2026-09-24 팀 결정). 모델은 캐시 키·예측 파일에 들어가므로 바꾸면 자동으로 다시 묻는다.
DEFAULT_PROVIDER = "nvidia"
# 2026-09-24 에 목록(`/v1/models`, 82개)에서 부를 수 있는 모델을 실제로 불러 골랐다. 목록에
# 있어도 404·503·60초 초과가 많았고, 답이 온 것은 이것과 `z-ai/glm-5.3` 이었다. 예비 레코드
# 5건에서 둘 다 형식 5/5, 사람 라벨과 4/5 로 같았고, 이쪽이 생각 과정을 끌 수 있어 평균
# 10초로 GLM(19초, 최대 43초, 생각 토큰 최대 1,547)보다 빠르고 답이 흔들릴 여지가 적다 (#59).
NVIDIA_MODEL = "deepseek-ai/deepseek-v4.1-flash"
# 이 모델은 기본으로 생각 과정(reasoning)을 먼저 쓴다. 그게 `MAX_ANSWER_TOKENS` 를 다 써 버리면
# 답(`content`)이 빈 채로 온다 - 실제 프롬프트 첫 호출이 그랬다. 모델을 바꿀 때 그 모델이 이
# 옵션을 무시하면(GLM 은 무시하고 늘 생각한다) 토큰 상한을 늘려야 한다.
NVIDIA_REQUEST_OPTIONS: dict[str, Any] = {"chat_template_kwargs": {"thinking": False}}
ANTHROPIC_MODEL = "claude-sonnet-5"
# Sonnet 5 는 `thinking` 을 빼면 생각 과정이 켜진 채로 돈다. 생각 토큰도 `max_tokens` 에 들어가서
# NVIDIA 에서 겪은 것처럼 답이 빈 채로 올 수 있고, 게이트 2 사전 등록의 "생각 과정 끔"과도 어긋난다.
# Opus 5.5·Fable 5.1 등은 `disabled` 를 400 으로 거부한다 - 모델을 바꾸면 이 값부터 다시 본다
# (Anthropic 문서 "Thinking", 2026-09-24 확인. 키가 없어 실제 호출로는 확인하지 못했다).
ANTHROPIC_REQUEST_OPTIONS: dict[str, Any] = {"thinking": {"type": "disabled"}}
DEFAULT_MODEL = NVIDIA_MODEL
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# 무료 한도가 분당 40회다. 넘으면 429 가 한 건의 호출 실패로 남아 그 레코드가 UNK 가 되므로,
# 한도보다 조금 느리게 부른다. 캐시에 있는 건은 호출기를 거치지 않아 기다리지 않는다.
NVIDIA_MIN_INTERVAL_SECONDS = 1.6
# 무료 한도는 대기열이 길 때가 있다 (실측 한 건 7~13초, 다른 모델은 60초를 넘기기도 했다).
NVIDIA_TIMEOUT_SECONDS = 120
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
MAX_ANSWER_TOKENS = 256
# 응답 캐시는 저장소에 커밋한다 (2026-09-24 팀 합의, `docs/evaluation.md` "LLM 모델 고정"). 무료
# 카탈로그에서 모델이 빠져도 같은 답으로 다시 채점할 수 있어야 한다. 그래서 `data/`(커밋하지 않음)
# 가 아니라 `datasets/` 아래가 기본이다. 파일에는 모델 이름과 답만 있고 키는 없다.
DEFAULT_CACHE_DIR = Path("datasets") / "llm_cache" / "baseline"
MAX_DIFF_CHARS = 4000
MAX_MESSAGE_CHARS = 2000

# §4.2 ③ 표를 그대로 옮긴다. 기준선이 우리와 **같은 분류 체계**로 답해야 비교가 성립한다.
LABEL_DEFINITIONS = (
    ("BUG", "잘못된 동작을 고치기 위해 제거"),
    ("PERF", "느리거나 자원을 많이 써서 교체"),
    ("SEC", "취약점·위험한 패턴 제거"),
    ("LIB", "직접 구현을 외부/표준 라이브러리로 대체"),
    ("DEAD", "호출되지 않아 제거"),
    ("DESIGN", "구조·추상화 변경으로 제거"),
    ("FEAT", "기능 자체를 없앰 (폐기, 지원 종료)"),
    ("UNK", "위 어느 것으로도 판단 불가"),
)

SYSTEM_PROMPT = (
    "당신은 오픈소스 커밋에서 함수가 삭제된 이유를 분류한다. "
    "주어진 정보만으로 판단하고, 없는 맥락을 지어내지 않는다. "
    "판단할 근거가 부족하면 UNK 를 고른다 - 억지로 고르는 것보다 낫다."
)

# 답을 한 줄로 받는다. 자유 서술을 파싱하면 실패가 늘고, 그 실패가 UNK 로 섞여 기준선을
# 실제보다 약하게 만든다.
ANSWER_RE = re.compile(r"^\s*([A-Z]+)\s*\|\s*(.*)$", re.MULTILINE)


def build_prompt(deleted_body: str, commit_message: str) -> str:
    """LLM 에 줄 본문. diff 와 커밋 메시지만 넣는다 - 기준선 B 의 정의가 "직접 질의" 다."""
    definitions = "\n".join(f"- {code}: {meaning}" for code, meaning in LABEL_DEFINITIONS)
    message = (commit_message or "(커밋 메시지 없음)")[:MAX_MESSAGE_CHARS]
    body = (deleted_body or "(삭제 코드 없음)")[:MAX_DIFF_CHARS]
    return (
        "아래 함수가 왜 삭제됐는지 한 가지로 분류하라.\n\n"
        f"분류 체계:\n{definitions}\n\n"
        f"커밋 메시지:\n{message}\n\n"
        f"삭제된 코드:\n```\n{body}\n```\n\n"
        "답은 정확히 한 줄로, `라벨|근거` 형식으로만 쓴다. "
        "라벨은 위 8개 중 하나이고, 근거는 한 문장이다.\n"
        "예: BUG|커밋 메시지에 빈 헤더에서 IndexError 가 났다고 적혀 있다"
    )


def parse_answer(text: str) -> tuple[str, str, str]:
    """LLM 응답 → (라벨, 근거, 실패 사유).

    8종 밖이거나 형식이 깨지면 UNK 로 두되 **사유를 남긴다.** 조용히 UNK 로 만들면 기준선이
    실제보다 약해 보이고, 우리 방식과의 차이가 실력이 아니라 파싱 실패 때문이 된다.
    """
    if not text or not text.strip():
        return UNKNOWN_LABEL, "", "빈 응답"

    match = ANSWER_RE.search(text)
    if match is None:
        snippet = text.strip().replace("\n", " ")[:80]
        return UNKNOWN_LABEL, "", f"형식 불일치: {snippet!r}"

    label, reason = match.group(1).strip().upper(), match.group(2).strip()
    if label not in REASON_LABELS:
        return UNKNOWN_LABEL, reason, f"8종 밖 라벨: {label!r}"
    return label, reason, ""


Caller = Callable[[str, str, str], str]
"""(system, prompt, model) -> 응답 텍스트. 테스트·다른 공급자를 위해 주입 가능하게 둔다."""

# 호출 한 건의 실패로 볼 예외. 여기 없는 예외는 배치 전체를 멈춘다.
# `http.client.HTTPException` 을 넣은 이유: 응답을 읽다 연결이 끊기면 `IncompleteRead` 가 나는데,
# 이건 `OSError` 가 아니라 `HTTPException` 계열이다. 빠져 있어서 기준선 B 와 분류기 LLM 후보가
# 둘 다 한 건의 끊김으로 멈췄다 (#84 코드래빗). 두 곳이 같은 목록을 쓰도록 여기 한 번만 둔다.
CALL_ERRORS: tuple[type[Exception], ...] = (
    urllib.error.URLError,
    OSError,
    ValueError,
    http.client.HTTPException,
)
# 응답 JSON 이 예상과 다른 모양일 때 답을 꺼내다 나는 예외들. 그대로 두면 `CALL_ERRORS` 밖이라
# 몇 시간짜리 배치가 한 건에서 멈추고, 예측 파일은 끝에 한 번에 쓰므로 아무것도 안 남는다.
# 호출기가 `ValueError` 로 바꿔 그 건만 실패로 남긴다 (#59 코드래빗).
SHAPE_ERRORS: tuple[type[Exception], ...] = (AttributeError, TypeError, KeyError, IndexError)


def anthropic_caller(api_key: str, *, timeout: int = 60) -> Caller:
    """Anthropic Messages API 호출기.

    SDK 를 의존성에 넣지 않고 urllib 으로 직접 부른다. 팀 전원이 패키지를 하나 더 설치해야
    하는 비용보다 POST 한 번을 직접 쓰는 편이 싸다 (select_repos 가 GitHub API 를 다루는
    방식과 같다).
    """

    def call(system: str, prompt: str, model: str) -> str:
        """한 번 묻고 답 텍스트를 돌려준다. 응답 모양이 다르면 `ValueError`."""
        payload = json.dumps(
            {
                "model": model,
                "max_tokens": MAX_ANSWER_TOKENS,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
                **ANTHROPIC_REQUEST_OPTIONS,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            ANTHROPIC_API_URL,
            data=payload,
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        try:
            blocks = body.get("content") or []
            return "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        except SHAPE_ERRORS as error:
            raise ValueError(f"응답 모양이 다르다: {error!r}") from error

    return call


def nvidia_caller(
    api_key: str,
    *,
    timeout: int = NVIDIA_TIMEOUT_SECONDS,
    min_interval: float = NVIDIA_MIN_INTERVAL_SECONDS,
) -> Caller:
    """NVIDIA API(build.nvidia.com) 호출기. OpenAI 호환 chat completions 다 (#59).

    `anthropic_caller` 와 같은 이유로 urllib 으로 직접 부른다. `temperature` 를 0 으로 두는 것은
    같은 입력에 같은 답을 받기 위해서다 - 캐시가 재실행을 막아 주지만, 캐시를 지운 사람이 다른
    답을 받으면 기준선 숫자를 재현할 수 없다.
    """
    last_call = float("-inf")

    def call(system: str, prompt: str, model: str) -> str:
        """간격을 지켜 한 번 묻는다. 응답 모양이 다르면 `ValueError`."""
        nonlocal last_call
        wait = last_call + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        last_call = time.monotonic()
        payload = json.dumps(
            {
                "model": model,
                "max_tokens": MAX_ANSWER_TOKENS,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                **NVIDIA_REQUEST_OPTIONS,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            NVIDIA_API_URL,
            data=payload,
            headers={"content-type": "application/json", "authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        try:
            choices = body.get("choices") or [{}]
            return (choices[0].get("message") or {}).get("content") or ""
        except SHAPE_ERRORS as error:
            raise ValueError(f"응답 모양이 다르다: {error!r}") from error

    return call


class Provider(NamedTuple):
    """LLM 공급자 하나. 키 이름·고정 모델·호출기를 한 곳에 묶는다."""

    env_key: str
    model: str
    make_caller: Callable[[str], Caller]


PROVIDERS: dict[str, Provider] = {
    "nvidia": Provider("NVIDIA_API_KEY", NVIDIA_MODEL, nvidia_caller),
    "anthropic": Provider("ANTHROPIC_API_KEY", ANTHROPIC_MODEL, anthropic_caller),
}


def caller_from_env(provider: str) -> Caller | None:
    """환경 변수의 키로 호출기를 만든다. 키가 없으면 알리고 `None` (§8.4 - 키는 `.env` 에만).

    기준선 B 와 분류기(`--llm`)가 같이 쓴다. 두 곳이 키 이름을 따로 들고 있으면 공급자를 바꿀
    때 한쪽만 바뀐다.
    """
    env_key = PROVIDERS[provider].env_key
    api_key = os.environ.get(env_key, "").strip()
    if not api_key:
        print(f"{env_key} 가 없다. .env 에 채워라 (§8.4).", file=sys.stderr)
        return None
    return PROVIDERS[provider].make_caller(api_key)


@dataclass
class LlmBaseline:
    """기준선 B 실행기. 호출기를 주입받아 네트워크 없이도 테스트할 수 있다."""

    caller: Caller
    model: str = DEFAULT_MODEL
    cache_dir: Path | None = None
    prompt_version: str = PROMPT_VERSION
    calls: int = 0
    cache_hits: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    def _cache_path(self, prompt: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = f"{self.prompt_version}|{self.model}|{prompt}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path) -> str | None:
        """캐시된 응답 텍스트. 없거나 쓸 수 없는 형태면 None (다시 묻는다).

        유효한 JSON 이어도 `text` 가 문자열이 아닐 수 있다 (손상·형식 변경). 그대로 넘기면
        `parse_answer` 의 문자열 연산에서 AttributeError 가 나고, 그건 `predict` 의 except 에
        걸리지 않아 배치 전체가 죽는다. 형태까지 확인하고서야 캐시로 인정한다.
        """
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        text = payload.get("text") if isinstance(payload, dict) else None
        return text if isinstance(text, str) else None

    def _write_cache(self, path: Path, text: str) -> None:
        """응답을 캐시에 남긴다. 실패해도 응답 자체는 버리지 않는다.

        이미 API 를 불러 받은 답이다. 디스크 문제로 그것을 UNK 로 만들면 돈을 쓰고도 결과를
        잃고, 실패 통계에도 "호출 실패" 로 잘못 기록돼 모델이 불안정한 것처럼 보인다.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"model": self.model, "text": text}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as error:
            self.failures["캐시 쓰기"] = self.failures.get("캐시 쓰기", 0) + 1
            print(f"캐시를 남기지 못했다 (응답은 그대로 쓴다): {error}", file=sys.stderr)

    def ask(self, prompt: str) -> str:
        """프롬프트 하나를 캐시를 거쳐 묻는다.

        공개로 둔 이유는 분류기(`classify.classifier`)의 LLM 후보 생성이 같은 캐시·호출기를
        쓰기 때문이다 (#84). 캐시 키에 프롬프트 버전이 들어가므로 프롬프트가 다르면 섞이지 않는다.
        """
        path = self._cache_path(prompt)
        if path is not None:
            cached = self._read_cache(path)
            if cached is not None:
                self.cache_hits += 1
                return cached

        self.calls += 1
        text = self.caller(SYSTEM_PROMPT, prompt, self.model)
        # 모든 호출기가 여기를 거치므로 답이 문자열인지는 여기서 한 번만 본다. OpenAI 호환 서버는
        # `content` 를 조각 리스트로 주기도 한다 - 그대로 두면 아래 `strip` 에서 배치가 멈춘다.
        if not isinstance(text, str):
            raise ValueError(f"응답이 문자열이 아니다: {type(text).__name__}")
        # 빈 답은 남기지 않는다. 답이 아니라 실패다 (생각 과정이 토큰을 다 쓰는 등) - 캐시에
        # 남기면 원인을 고친 뒤 다시 돌려도 그 건은 영영 다시 묻지 않는다 (#59 에서 실제로 그랬다).
        if path is not None and text.strip():
            self._write_cache(path, text)
        return text

    def predict(self, record: dict[str, Any]) -> Prediction:
        """레코드 하나를 분류한다. 호출이 실패해도 예측을 돌려준다 (사유를 남기고 UNK)."""
        prompt = build_prompt(record.get("deleted_body", ""), commit_message_of(record))
        version = f"{self.prompt_version}/{self.model}"

        try:
            text = self.ask(prompt)
        except CALL_ERRORS as error:
            self.failures["호출 실패"] = self.failures.get("호출 실패", 0) + 1
            return Prediction(
                record_id=record_id_of(record),
                predicted_label=UNKNOWN_LABEL,
                method=METHOD_LLM,
                version=version,
                note=f"호출 실패: {type(error).__name__}: {error}",
            )

        label, reason, problem = parse_answer(text)
        if problem:
            self.failures[problem.split(":")[0]] = self.failures.get(problem.split(":")[0], 0) + 1
        return Prediction(
            record_id=record_id_of(record),
            predicted_label=label,
            method=METHOD_LLM,
            version=version,
            evidence=(reason,) if reason else (),
            note=problem,
        )

    def predict_all(self, records: Sequence[dict[str, Any]]) -> list[Prediction]:
        return [self.predict(record) for record in records]


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자. 실행 방법은 모듈 독스트링 "실행" 절."""
    parser = argparse.ArgumentParser(
        prog="python -m classify.baseline_llm",
        description="기준선 B - LLM 에 diff + 커밋 메시지를 주고 이유 8종 예측 (#44).",
    )
    parser.add_argument("--input", type=Path, required=True, help="레코드 JSONL")
    parser.add_argument("--out", type=Path, default=None, help="예측 JSONL 저장 경로")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=None, help="기본: 공급자별 고정 모델")
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N건만 (비용 확인용)")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--dry-run", action="store_true", help="API 를 부르지 않고 첫 프롬프트만 출력"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """기준선 B 를 돌려 예측을 저장하고 분포·호출 수·실패를 보고한다."""
    args = build_parser().parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    load_env_file(args.env_file)

    records = read_jsonl(args.input)
    # 음수를 그대로 슬라이스하면 records[:-1] 이 되어 "앞 N건만" 의 정반대가 된다.
    # 마지막 한 건만 빼고 전부 유료 호출하는 셈이라, 비용을 아끼려는 옵션이 비용을 쓴다.
    if args.limit < 0:
        print("--limit 는 0 이상이어야 한다 (0 이면 전체).", file=sys.stderr)
        return 2
    if args.limit:
        records = records[: args.limit]
    if not records:
        print("레코드가 없다.", file=sys.stderr)
        return 1

    if args.dry_run:
        first = records[0]
        print(build_prompt(first.get("deleted_body", ""), commit_message_of(first)))
        print(f"\n(dry-run: {len(records)}건 대상, API 를 부르지 않았다)", file=sys.stderr)
        return 0

    caller = caller_from_env(args.provider)
    if caller is None:
        return 2

    cache_dir = args.cache_dir or DEFAULT_CACHE_DIR
    model = args.model or PROVIDERS[args.provider].model
    baseline = LlmBaseline(caller, model=model, cache_dir=cache_dir)
    predictions = baseline.predict_all(records)

    if args.out:
        written = write_predictions(args.out, predictions)
        print(f"예측: {args.out} ({written}건)", file=sys.stderr)

    print(f"기준선 B (프롬프트 {baseline.prompt_version} / 모델 {baseline.model})")
    for line in summarize(predictions).format_lines():
        print(line)
    print(f"  API {baseline.calls}회 / 캐시 {baseline.cache_hits}회")
    if baseline.failures:
        detail = ", ".join(f"{reason} {count}건" for reason, count in baseline.failures.items())
        print(f"  실패: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
