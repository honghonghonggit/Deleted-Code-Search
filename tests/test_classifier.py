"""분류기 골격 테스트 (이슈 #84) - 규칙·모델·LLM 후보·앙상블·근거 등급.

네트워크를 타지 않는다. LLM 은 가짜 호출기로 본다. 정확도는 보지 않는다 - 끝까지 연결되는지와
근거 규칙(가이드 §6)을 지키는지만 본다.
"""

import http.client
import json

import pytest

from classify import classifier as clf
from classify import rules
from classify.baseline_llm import LlmBaseline
from classify.model import ReasonModel
from classify.model import record_text as model_text


def make_record(record_id="r1", **overrides):
    """§4.4 레코드 모양의 최소 레코드. 맥락은 비어 있다 - 테스트마다 채운다."""
    record = {
        "id": record_id,
        "repo": "a/b",
        "file_path": "src/net/retry_helper.py",
        "function_name": "legacy_backoff",
        "deleted_body": "def legacy_backoff():\n    return 1\n",
        "context": {
            "commit_message": "",
            "pr_number": None,
            "pr_title": None,
            "pr_body": None,
            "pr_labels": [],
            "issue_numbers": [],
            "issue_titles": [],
            "issue_bodies": [],
            "review_comments": [],
        },
        "replacement": {"code": None, "match_method": None, "confidence": 0.0},
    }
    for key, value in overrides.items():
        if key in record["context"]:
            record["context"][key] = value
        else:
            record[key] = value
    return record


def fake_llm(answer):
    """늘 같은 답을 주는 LLM 후보."""
    return clf.LlmCandidate(LlmBaseline(caller=lambda system, prompt, model: answer))


class FixedModel:
    """늘 같은 확률을 내는 모델 - 점수 합산과 동점 규칙만 보려고 쓴다."""

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, record):
        return dict(self.probabilities)


def trained_model(label="DESIGN"):
    """`layering` 이 나오면 `label`, `segfault` 가 나오면 BUG 로 기우는 모델.

    두 단어는 기준선 A 키워드가 **아니다** - 규칙은 아무것도 못 찾고 모델만 답하는 상황을
    만들려고 골랐다.
    """
    records, labels = [], []
    for index in range(3):
        for reason, word in ((label, "layering"), ("BUG", "segfault")):
            records.append(make_record(f"{reason}{index}", commit_message=f"{word} {word}"))
            labels.append(reason)
    return ReasonModel().fit(records, labels)


# --------------------------------------------------------------------------------------
# 규칙 - 문장·위치 (가이드 §7.2)
# --------------------------------------------------------------------------------------


def test_every_context_source_gets_its_locator():
    """맥락의 출처마다 가이드 §7.2 형식의 위치가 붙는다."""
    record = make_record(
        commit_message="Remove the retry helper.",
        pr_number=12,
        pr_title="Drop legacy retry",
        pr_body="It raced on reconnect.",
        issue_numbers=[7],
        issue_titles=["Backoff races"],
        issue_bodies=["Two threads double the delay."],
        review_comments=[
            {"comment_id": 99, "body": "Use urllib3 Retry instead."},
            {"comment_id": None, "body": "Why was this still here?"},
        ],
    )
    # 문장과 위치를 **묶어서** 본다. 위치 집합만 보면 제목·본문 로케이터가 뒤바뀌어도 통과한다.
    found = {(p.text, p.source, p.locator) for p in rules.passages(record)}

    assert found == {
        ("Remove the retry helper.", "commit", "commit:message"),
        ("Drop legacy retry", "pr", "pr:#12#title"),
        ("It raced on reconnect.", "pr", "pr:#12#body"),
        ("Backoff races", "issue", "issue:#7#title"),
        ("Two threads double the delay.", "issue", "issue:#7#body"),
        ("Use urllib3 Retry instead.", "review", "review:comment_99"),
        ("Why was this still here?", "review", "review:unknown"),
    }


def test_issue_lists_of_different_lengths_are_not_paired():
    """번호·제목 길이가 다르면 앞에서부터 짝을 짓지 않는다 - 제목이 엉뚱한 번호에 붙는다."""
    record = make_record(issue_numbers=[7, 8], issue_titles=["Title that belongs to issue 8"])

    assert rules.passages(record) == []


def test_old_string_review_comments_still_read_as_unknown_locator():
    """ADR-018 전 레코드는 리뷰 코멘트가 문자열이다. 터지지 않고 `review:unknown` 으로 읽는다."""
    record = make_record(review_comments=["this was never called anywhere"])

    assert [p.locator for p in rules.passages(record)] == ["review:unknown"]


def test_code_blocks_are_not_read_as_reasons():
    """코드 예시 안의 단어가 이유 키워드로 잡히면 안 된다 (`pipeline.context.strip_code`)."""
    record = make_record(commit_message="Cleanup\n\n```\n# fix the crash later\n```")

    assert all("crash" not in p.text for p in rules.passages(record))


def test_pr_template_comments_are_not_read():
    """PR 템플릿 안내문은 작성자 글이 아니다. pydantic 템플릿의 `"fix #123"` 이 BUG 로 잡혔다."""
    record = make_record(
        pr_number=12,
        pr_body='<!-- WARNING: please use "fix #123" style references -->\nRemove unused helper.',
    )

    assert [p.text for p in rules.passages(record)] == ["Remove unused helper."]


def test_an_issue_closing_reference_alone_is_not_a_reason():
    """`Fixes #12` 는 "이 이슈를 닫는다" 는 연결이다. 이슈가 기능 요청이어도 BUG 가 되면 안 된다."""
    for message in ("Fixes #3409.", "Fix GH-12", "closes https://github.com/a/b/issues/7"):
        assert rules.find_reason_sentences(make_record(commit_message=message)) == []


def test_closing_reference_is_ignored_for_the_label_but_kept_in_the_quote():
    """라벨은 참조를 빼고 고르되, 인용문은 원문 그대로다 (가이드 §6.1)."""
    record = make_record(commit_message="Remove unused legacy_backoff, fixes #12")
    [sentence] = rules.find_reason_sentences(record)

    assert sentence.label == "DEAD"
    assert sentence.passage.text == "Remove unused legacy_backoff, fixes #12"


def test_markdown_bullets_are_stripped_and_fragments_dropped():
    """목록 기호는 벗기고 체크박스 잔해 같은 짧은 조각은 버린다."""
    sentences = rules.split_sentences("- [x]\n* Removed unused helper\n## Why")

    assert sentences == ["Removed unused helper"]


def test_target_name_must_be_a_whole_word():
    """`parse` 가 `parser` 에 걸리면 "이 함수를 가리킨다" 가 거짓이 된다."""
    assert rules.mentions("drop parse_all", ("parse_all",))
    assert not rules.mentions("drop parse_all_items", ("parse_all",))


def test_inline_code_stays_in_the_quote_and_a_backticked_name_is_seen():
    """인라인 코드를 지우면 인용문이 원문과 달라지고, 백틱 안 함수 이름을 못 본다.

    예비 200건에서 174건 중 63건의 인용문이 원문에 없었다 (``Support `Field(repr=False)` in``
    이 ``Support   in`` 이 됐다). 함수 이름은 보통 백틱 안에 쓴다.
    """
    message = "Remove unused `legacy_backoff`."
    result = clf.Classifier(llm=fake_llm("DEAD|x")).classify(make_record(commit_message=message))

    assert result.evidence_text == message
    assert (result.label, result.evidence_grade) == ("DEAD", "EXPLICIT")


def test_words_inside_inline_code_do_not_pick_the_label():
    """`fix_headers` 같은 식별자 안의 `fix` 가 BUG 로 잡히면 안 된다."""
    assert rules.find_reason_sentences(make_record(commit_message="Rename `fix_headers`.")) == []


def test_file_name_is_not_a_target():
    """함수만 지워졌고 파일은 남았다 - "삭제된 파일" 이 아니다 (가이드 §6.1.1 E2-(가)).

    파일 이름을 대상으로 두었을 때 EXPLICIT 10건 중 9건이 파일 이름으로만 나왔고 틀렸다.
    """
    record = make_record(
        function_name="dataclass",
        file_path="pydantic/dataclasses.py",
        commit_message="fix dataclasses and docs",
    )

    assert rules.target_names(record) == ("dataclass",)
    assert clf.Classifier(llm=fake_llm("BUG|x")).classify(record).evidence_grade == "INFERRED"


@pytest.mark.parametrize("message", ["Update docs to fix typo.", "update docs to fix typo."])
def test_a_plain_word_function_name_in_prose_is_not_explicit(message):
    """`update` 는 영어 단어와 모양이 같다. 문장에 나왔다고 그 함수를 가리키는 게 아니다.

    예비 200건에서 `host` 함수가 "Fix host required enforcement ..." 로 EXPLICIT 1.0 이 됐다.
    """
    record = make_record(function_name="update", commit_message=message)
    result = clf.Classifier(llm=fake_llm("BUG|x")).classify(record)

    assert result.label == "BUG"
    assert result.evidence_grade != "EXPLICIT"


@pytest.mark.parametrize(
    "message",
    ["Remove unused `update`.", "Remove unused `Model.update`.", "Remove unused update()."],
)
def test_a_plain_word_function_name_counts_with_a_code_marker(message):
    """백틱 안이거나 괄호가 붙으면 코드를 가리킨 것이다."""
    record = make_record(function_name="update", commit_message=message)

    assert clf.Classifier(llm=fake_llm("DEAD|x")).classify(record).evidence_grade == "EXPLICIT"


def test_identifier_names_are_matched_case_sensitively():
    """식별자는 대소문자를 가린다. `LEGACY_BACKOFF` 는 다른 이름이다."""
    assert rules.mentions("drop legacy_backoff", ("legacy_backoff",))
    assert not rules.mentions("drop LEGACY_BACKOFF", ("legacy_backoff",))


def test_dunder_names_are_not_targets():
    """`__init__` 은 클래스마다 있어 이름만으로 어느 함수인지 가리키지 못한다."""
    assert rules.target_names(make_record(function_name="__init__")) == ()


def test_short_function_names_are_not_used_as_targets():
    """`get`·`run` 같은 이름은 아무 문장에나 걸린다."""
    assert rules.target_names(make_record(function_name="get", file_path="x/io.py")) == ()


def test_reason_sentences_carry_label_and_whether_they_name_the_target():
    """이유 문장마다 라벨과 함수를 이름으로 가리키는지가 붙는다."""
    record = make_record(
        commit_message="legacy_backoff is unused now. Also fix typo in docs.",
    )
    found = {(s.label, s.names_target) for s in rules.find_reason_sentences(record)}

    assert found == {("DEAD", True), ("BUG", False)}


# --------------------------------------------------------------------------------------
# 분류 - 모델 확률 + LLM 이 고르고, 규칙은 등급과 인용 (#86)
# --------------------------------------------------------------------------------------


def test_sentence_keywords_do_not_vote_so_unk_can_win():
    """키워드 문장(`unused` = DEAD)이 있어도 표를 던지지 않는다 - UNK 도 점수로 겨룬다.

    #84 는 키워드 문장이 있으면 UNK 가 될 수 없었다. 학습 300건 교차 검증에서 키워드 표가 BUG 를
    98건(정답 6건) 냈고, 사람 UNK 135건 중 79건이 그 관문을 통과했다 (`classify.model`).
    """
    result = clf.Classifier(llm=fake_llm("UNK|이 함수를 가리키는 근거가 없다")).classify(
        make_record(commit_message="Remove unused helpers.")
    )

    assert (result.label, result.evidence_grade, result.confidence) == ("UNK", "UNKNOWN", 0.0)


def test_llm_vote_competes_with_the_model_probability():
    """LLM 이 고른 라벨에는 `WEIGHT_LLM` 이 더해진다. 모델이 더 확신하면 모델 쪽이 이긴다."""
    record = make_record(commit_message="Adjust layering")
    llm = fake_llm("BUG|x")
    sure = clf.Classifier(model=FixedModel({"DESIGN": clf.WEIGHT_LLM + 0.3, "BUG": 0.1}), llm=llm)
    unsure = clf.Classifier(model=FixedModel({"DESIGN": 0.55, "BUG": 0.45}), llm=llm)

    assert sure.classify(record).label == "DESIGN"
    assert unsure.classify(record).label == "BUG"


def test_sentence_naming_the_function_is_explicit_with_its_locator():
    """같은 라벨 문장이 함수를 이름으로 가리키면 EXPLICIT - 원문과 위치가 그대로 남는다."""
    record = make_record(pr_number=12, pr_body="legacy_backoff is no longer used by the client.")
    result = clf.Classifier(llm=fake_llm("DEAD|x")).classify(record)

    assert (result.label, result.evidence_grade) == ("DEAD", "EXPLICIT")
    assert result.confidence == clf.EXPLICIT_CONFIDENCE
    assert result.evidence_locator == "pr:#12#body"
    assert result.evidence_text == "legacy_backoff is no longer used by the client."


def test_sentence_that_does_not_reach_this_function_is_inferred():
    """이유는 말하지만 이 함수를 가리키지 않는다 - 가이드 §6.1.1 E2 를 못 넘는다."""
    record = make_record(commit_message="Remove unused helpers across the package.")
    result = clf.Classifier(llm=fake_llm("DEAD|x")).classify(record)

    assert (result.label, result.evidence_grade) == ("DEAD", "INFERRED")
    assert result.confidence == clf.UNREACHED_SENTENCE_CONFIDENCE
    assert result.evidence_locator == "commit:message"


def test_replacement_alone_is_inferred_evidence_one():
    """대체 코드만 있으면 근거 ① - `diff:replacement`, 이유는 모델이 고른다."""
    record = make_record(
        commit_message="Adjust layering",
        replacement={
            "code": "def backoff():\n    ...",
            "match_method": "SAME_LOCATION",
            "confidence": 0.9,
        },
    )
    result = clf.Classifier(model=trained_model("DESIGN")).classify(record)

    assert (result.label, result.evidence_grade) == ("DESIGN", "INFERRED")
    assert (result.evidence_source, result.evidence_locator) == ("diff", "diff:replacement")
    assert result.confidence == 0.9


def test_replacement_confidence_caps_the_inferred_confidence():
    """파이프라인이 0.7 만 확신하는 대체 코드로 0.8+ 를 줄 수 없다 (#103 0.7 논의)."""
    record = make_record(
        replacement={
            "code": "def backoff(): ...",
            "match_method": "SAME_LOCATION",
            "confidence": 0.7,
        },
    )

    assert clf.Classifier(model=trained_model()).classify(record).confidence == 0.7


def test_replacement_confidence_above_the_cap_is_cut_to_0_9():
    """INFERRED 는 1.0 을 쓰지 않는다 (가이드 §6.2.2). 대체 코드 신뢰도가 1.0 이어도 0.9 다."""
    record = make_record(
        replacement={
            "code": "def backoff(): ...",
            "match_method": "SAME_LOCATION",
            "confidence": 1.0,
        },
    )

    assert clf.Classifier(model=trained_model()).classify(record).confidence == 0.9


def test_replacement_exactly_at_the_floor_is_still_inferred():
    """0.5 는 하한 안쪽이다 - 0.5 미만부터 UNKNOWN (가이드 §6.2.2)."""
    record = make_record(
        replacement={
            "code": "def backoff(): ...",
            "match_method": "SAME_LOCATION",
            "confidence": 0.5,
        },
    )
    result = clf.Classifier(model=trained_model()).classify(record)

    assert (result.evidence_grade, result.confidence) == ("INFERRED", 0.5)


@pytest.mark.parametrize(
    ("code", "confidence"),
    [
        ("def backoff(): ...", float("nan")),  # json.loads 가 NaN 을 받는다
        ("def backoff(): ...", True),  # bool 은 int 라 1.0 이 된다
        ("   \n  ", 0.9),  # 공백뿐인 코드
    ],
)
def test_malformed_replacement_is_not_evidence(code, confidence):
    """`min(0.9, nan)` 은 0.9 다 - 확인 없이 두면 근거 없는 0.9 가 새어 나간다."""
    record = make_record(
        replacement={"code": code, "match_method": "SAME_LOCATION", "confidence": confidence}
    )
    result = clf.Classifier(model=trained_model()).classify(record)

    assert result.evidence_locator != "diff:replacement"
    assert result.confidence == clf.INFERRED_FLOOR


def test_replacement_with_no_signal_for_any_reason_is_unk_not_sec():
    """대체 코드만 있고 규칙·모델·LLM 이 모두 0 이면 이유를 말할 신호가 없다.

    확인하지 않으면 7종이 모두 0 인 동점에서 §11-1 첫 순위 SEC 가 0.9 로 나온다. 모델이
    학습되지 않은 채(`--records` 에 라벨과 이어지는 건이 없을 때) 돌리면 실제로 그렇게 된다.
    """
    record = make_record(
        replacement={
            "code": "def backoff(): ...",
            "match_method": "SAME_LOCATION",
            "confidence": 0.9,
        },
    )
    result = clf.Classifier().classify(record)

    assert (result.label, result.evidence_grade, result.confidence) == ("UNK", "UNKNOWN", 0.0)


def test_explicit_comes_only_from_a_sentence_of_the_chosen_label():
    """함수를 이름으로 가리키는 문장이 있어도, 그 문장이 **다른 이유**를 말하면 EXPLICIT 이 아니다.

    인용문이 고른 이유를 말해야 한다 (가이드 §6.1). 여기서는 DEAD 가 골라지는데, 함수 이름이
    든 문장은 DESIGN 을 말한다.
    """
    record = make_record(
        commit_message="Refactor legacy_backoff. Remove unused code. Drop obsolete helpers."
    )
    result = clf.Classifier(llm=fake_llm("DEAD|x")).classify(record)

    assert result.label == "DEAD"
    assert result.evidence_grade == "INFERRED"
    assert "legacy_backoff" not in result.evidence_text


def test_replacement_below_the_inferred_floor_is_not_evidence():
    """0.5 아래는 INFERRED 가 아니라 UNKNOWN 이다 (가이드 §6.2.2)."""
    record = make_record(
        replacement={
            "code": "def backoff(): ...",
            "match_method": "SAME_LOCATION",
            "confidence": 0.3,
        },
    )

    assert clf.Classifier(model=trained_model()).classify(record).evidence_locator == ""


def test_null_replacement_code_is_not_evidence():
    """`code` 가 `null` 이면 근거 ① 을 쓸 수 없다 (가이드 §6.2.3)."""
    record = make_record(
        replacement={"code": None, "match_method": "SAME_LOCATION", "confidence": 0.9}
    )

    assert clf.Classifier(model=trained_model()).classify(record).evidence_locator == ""


# --------------------------------------------------------------------------------------
# LLM - 후보와 근거 문장만 (ADR-005)
# --------------------------------------------------------------------------------------


def test_on_a_tie_the_label_backed_by_evidence_wins():
    """점수가 같으면 이유 문장이 받치는 라벨이 이긴다 (문장은 DEAD, 우선순위로는 SEC 가 앞)."""
    record = make_record(commit_message="Remove unused helpers.")
    result = clf.Classifier(model=FixedModel({"SEC": 0.5, "DEAD": 0.5})).classify(record)

    assert result.label == "DEAD"


def test_model_can_pick_a_reason_the_sentence_keyword_missed():
    """문장 키워드가 다른 이유를 가리켜도 모델이 강하면 그 이유를 고를 수 있다.

    예비 200건에서 사람 DEAD 23건의 문장 키워드는 대부분 DESIGN 이었다. 키워드 라벨로 후보를
    자르면 DEAD 가 아예 나올 수 없었다. 이때 등급은 가장 약한 INFERRED(하한) 다.

    #86 부터는 키워드 문장이 표를 던지지 않아, 이것은 등급을 보는 테스트다 - 고른 라벨의 문장이
    없으니 가장 약한 INFERRED 이고, 어느 문장이 다른 이유를 가리켰는지 note 에 남는다.
    """
    # 문장 키워드는 DESIGN(tidy)·PERF(faster) 로 갈리고, 둘 다 모델이 배우지 않은 이유다.
    record = make_record(commit_message="Tidy things. Make it faster. more layering layering")
    result = clf.Classifier(model=trained_model("DEAD")).classify(record)

    assert result.label == "DEAD"
    assert (result.evidence_grade, result.confidence) == ("INFERRED", clf.INFERRED_FLOOR)
    assert "DESIGN" in result.note


def test_without_model_or_llm_there_is_no_signal_and_it_is_unk():
    """학습되지 않은 모델 + LLM 없음이면 이유를 말할 신호가 없다 - 키워드 문장이 있어도 UNK."""
    result = clf.Classifier().classify(make_record(commit_message="Remove unused code."))

    assert (result.label, result.evidence_grade, result.note) == ("UNK", "UNKNOWN", "신호 없음")


def test_llm_reason_is_the_evidence_when_only_the_llm_backs_the_label():
    """고른 라벨의 문장도 대체 코드도 없으면 LLM 이 화면을 보고 쓴 근거를 INFERRED 하한으로 쓴다."""
    reason = "추가 헝크 [2/3] 이 같은 일을 새 클래스에서 한다"
    record = make_record(commit_message="Adjust layering")
    result = clf.Classifier(llm=fake_llm(f"DESIGN|{reason}")).classify(record)

    assert (result.label, result.evidence_grade, result.evidence_text) == (
        "DESIGN",
        "INFERRED",
        reason,
    )
    assert (result.confidence, result.evidence_locator) == (clf.INFERRED_FLOOR, "")
    assert "LLM" in result.note


def test_llm_writes_the_evidence_sentence_only_for_inferred_from_replacement():
    """EXPLICIT 은 원문 인용이어야 해서 LLM 이 쓰지 않는다. 대체 코드 근거는 써 줄 수 있다."""
    record = make_record(
        replacement={
            "code": "def backoff(): ...",
            "match_method": "SAME_LOCATION",
            "confidence": 0.9,
        },
    )
    result = clf.Classifier(llm=fake_llm("LIB|urllib3 Retry 로 바꿨다")).classify(record)

    assert (result.label, result.evidence_text) == ("LIB", "urllib3 Retry 로 바꿨다")
    assert "LLM" in result.note


def test_llm_never_replaces_an_explicit_quote():
    """LLM 이 같은 이유를 골라도 EXPLICIT 인용문은 원문 그대로다 (가이드 §6.1, ADR-005)."""
    record = make_record(commit_message="legacy_backoff is unused now.")
    result = clf.Classifier(llm=fake_llm("DEAD|아무도 부르지 않는 함수라 지웠다")).classify(record)

    assert result.evidence_grade == "EXPLICIT"
    assert result.evidence_text == "legacy_backoff is unused now."


def test_version_says_whether_and_which_llm_was_used():
    """LLM 유무·모델이 다른 두 예측 파일이 같은 버전을 달면 가를 수 없다."""
    with_llm = clf.Classifier(llm=fake_llm("DEAD|x"))

    assert clf.Classifier().version == clf.CLASSIFIER_VERSION
    assert with_llm.version == f"{clf.CLASSIFIER_VERSION}+llm:{with_llm.llm.runner.model}"
    assert with_llm.classify(make_record(commit_message="Remove unused.")).version == (
        with_llm.version
    )


def test_llm_failure_does_not_stop_classification():
    """LLM 호출이 실패해도 표만 빠지고 분류는 계속된다."""

    def broken(system, prompt, model):
        """늘 네트워크 오류를 내는 호출기."""
        raise OSError("network down")

    candidate = clf.LlmCandidate(LlmBaseline(caller=broken))
    classifier = clf.Classifier(model=trained_model("DEAD"), llm=candidate)
    result = classifier.classify(make_record(commit_message="layering layering"))

    assert result.label == "DEAD"  # 모델만으로 고른다
    assert candidate.failures == {"호출 실패: OSError": 1}


def test_connection_dropped_mid_response_does_not_stop_classification():
    """`IncompleteRead` 는 `OSError` 가 아니다 - 빠져 있어 한 건의 끊김으로 배치가 멈췄다."""

    def dropped(system, prompt, model):
        """응답을 읽다 연결이 끊긴 호출기."""
        raise http.client.IncompleteRead(b"partial")

    candidate = clf.LlmCandidate(LlmBaseline(caller=dropped))
    classifier = clf.Classifier(model=trained_model("DEAD"), llm=candidate)
    result = classifier.classify(make_record(commit_message="layering layering"))

    assert result.label == "DEAD"
    assert candidate.failures == {"호출 실패: IncompleteRead": 1}


def test_llm_prompt_carries_context_locators_and_replacement():
    """기준선 B 와 달리 맥락 전체와 대체 코드를 준다 - 그 차이가 우리 방식이다."""
    record = make_record(
        pr_number=12,
        pr_body="Switched to urllib3 Retry.",
        replacement={
            "code": "def backoff(): ...",
            "match_method": "SAME_LOCATION",
            "confidence": 0.9,
        },
    )
    prompt = clf.build_candidate_prompt(record)

    assert "[pr:#12#body] Switched to urllib3 Retry." in prompt
    assert "def backoff(): ..." in prompt


def test_llm_prompt_shows_the_labeler_screen_and_the_guide():
    """c2 부터: 파일·함수·테스트 여부, 같은 파일 추가 헝크, 가이드 판정 순서를 준다 (#86)."""
    record = make_record(
        is_test_code=True,
        added_hunks_same_file=[
            {"new_start": 7, "added_body": "    return retry(1)"},
            {"new_start": 20, "added_body": "def retry(n):"},
        ],
    )
    prompt = clf.build_candidate_prompt(record)

    assert "파일: src/net/retry_helper.py" in prompt
    assert "함수: legacy_backoff" in prompt
    assert "테스트 코드: 예" in prompt
    assert "[1/2] new_start 7\n    return retry(1)\n[2/2] new_start 20\ndef retry(n):" in prompt
    assert clf.CANDIDATE_GUIDE in prompt


def test_long_added_hunks_are_cut():
    """추가 헝크 상위 10% 는 6천 자를 넘는다 - 프롬프트를 다 차지하지 않게 자른다."""
    record = make_record(added_hunks_same_file=[{"new_start": 1, "added_body": "x" * 10_000}])
    text = clf.added_hunks_text(record)

    assert len(text) < clf.MAX_HUNKS_CHARS + 50
    assert text.endswith("(이하 생략)")
    assert clf.added_hunks_text(make_record()) == "(이 커밋이 이 파일에 추가한 줄 없음)"


# --------------------------------------------------------------------------------------
# 출력 형식
# --------------------------------------------------------------------------------------


def test_schema_reason_has_exactly_the_charter_fields():
    """§4.4 `reason` 과 이름·개수가 같아야 한다 (§13). 로케이터는 칸이 없어 넣지 않는다."""
    result = clf.Classifier().classify(make_record(commit_message="Remove unused."))

    assert set(result.to_schema_reason()) == {
        "label",
        "evidence_grade",
        "evidence_text",
        "confidence",
        "classifier_version",
    }


def test_prediction_file_uses_predicted_label_not_reason_label():
    """사람 라벨(`reason_label`)과 이름을 갈라 섞이면 바로 보이게 한다 (ADR-005)."""
    classifier = clf.Classifier(llm=fake_llm("DEAD|x"))
    row = classifier.classify(make_record(commit_message="Remove unused.")).to_dict()

    assert row["predicted_label"] == "DEAD"
    assert "reason_label" not in row
    assert row["method"] == clf.METHOD_OURS


def test_classification_rejects_labels_outside_the_eight():
    """§4.2 ③ 8종 밖의 라벨로는 결과를 만들 수 없다."""
    with pytest.raises(ValueError):
        clf.Classification("r", "MAYBE", "EXPLICIT", "", "", "", 1.0)


# --------------------------------------------------------------------------------------
# 모델·입력
# --------------------------------------------------------------------------------------


def test_model_gives_probabilities_over_the_reasons_it_saw():
    """모델은 학습에 나온 이유들에 대해서만 확률을 낸다."""
    model = trained_model("DESIGN")
    probabilities = model.predict_proba(make_record(commit_message="more layering"))

    assert set(probabilities) == {"BUG", "DESIGN"}
    assert probabilities["DESIGN"] > probabilities["BUG"]


def test_model_learns_unk_with_the_screen_shape():
    """m2 는 UNK 도 배운다. 맥락이 없는 모양(`zznopr` 등)이 UNK 쪽 신호가 된다 (가이드 §6.3.1)."""
    bare = [make_record(f"u{i}", commit_message="wip") for i in range(3)]
    rich = [
        make_record(f"d{i}", commit_message="drop it", pr_number=i, pr_body="unused helper")
        for i in range(3)
    ]
    model = ReasonModel().fit(bare + rich, ["UNK"] * 3 + ["DEAD"] * 3)

    assert "UNK" in model.classes
    probabilities = model.predict_proba(make_record("new", commit_message="wip"))
    assert probabilities["UNK"] > probabilities["DEAD"]
    assert "zznopr" in model_text(bare[0])
    assert "zzhaspr" in model_text(rich[0])


def test_model_with_a_single_reason_stays_untrained_instead_of_crashing():
    """이유가 한 종류뿐이면 학습하지 않고 빈 확률을 낸다 - 예외로 멈추지 않는다."""
    records = [make_record(f"t{i}") for i in range(3)]
    model = ReasonModel().fit(records, ["DEAD", "DEAD", "DEAD"])

    assert not model.trained
    assert model.predict_proba(records[0]) == {}


def test_only_settled_labels_are_used_for_training():
    """두 사람이 갈려 확정되지 않은 건을 한쪽 라벨로 채우면 그 사람 판단을 정답으로 배운다."""
    rows = [
        {"record_id": "a", "final": {"reason_label": "DEAD"}},
        {"record_id": "b", "final": {"reason_label": None}},
        {"record_id": "c", "final": {}},
    ]

    assert clf.load_final_labels(rows) == {"a": "DEAD"}


def test_cli_runs_end_to_end_without_an_api_key(tmp_path, capsys):
    """API 키 없이 CLI 가 학습·분류·저장까지 끝까지 돈다."""
    records = [
        make_record("a", commit_message="Remove unused helpers."),
        make_record("b", commit_message="Refactor retry module."),
        make_record("c", commit_message="Update things"),
    ]
    labels = [
        {"record_id": "a", "final": {"reason_label": "DEAD"}},
        {"record_id": "b", "final": {"reason_label": "DESIGN"}},
    ]
    records_path, labels_path = tmp_path / "records.jsonl", tmp_path / "labels.jsonl"
    out_path = tmp_path / "out.jsonl"
    records_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    labels_path.write_text("\n".join(json.dumps(r) for r in labels), encoding="utf-8")

    code = clf.main(
        ["--records", str(records_path), "--labels", str(labels_path), "--out", str(out_path)]
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert code == 0
    assert [row["predicted_label"] for row in rows][:2] == ["DEAD", "DESIGN"]
    assert len(rows) == 3
    printed = capsys.readouterr().out
    assert "정확도를 말하지 않는다" in printed
    # 읽은 라벨 수가 아니라 레코드와 실제로 이어진 수를 말해야 한다.
    assert "레코드와 이어진 2건" in printed


def test_cli_llm_without_key_names_the_default_provider_key(tmp_path, monkeypatch, capsys):
    """`--llm` 은 기준선 B 와 같은 공급자 표를 쓴다 - 기본은 NVIDIA (#59)."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    records_path, labels_path = tmp_path / "records.jsonl", tmp_path / "labels.jsonl"
    records_path.write_text(json.dumps(make_record("a")), encoding="utf-8")
    labels_path.write_text("", encoding="utf-8")

    code = clf.main(
        ["--records", str(records_path), "--labels", str(labels_path), "--llm"]
        + ["--env-file", str(tmp_path / "none")]
    )

    assert code == 2
    assert "NVIDIA_API_KEY" in capsys.readouterr().err


def split_files(tmp_path):
    """학습 2건(t1 DEAD, t2 DESIGN) + 검증 1건(v1) + 시험 1건(x1)."""
    records = [
        make_record("t1", commit_message="unused unused"),
        make_record("t2", commit_message="layering layering"),
        make_record("v1", commit_message="unused unused here"),
        make_record("x1", commit_message="layering layering here"),
    ]
    labels = [
        {"record_id": "t1", "split": "train", "final": {"reason_label": "DEAD"}},
        {"record_id": "t2", "split": "train", "final": {"reason_label": "DESIGN"}},
        # 검증 건의 정답이 학습에 쓰이면 v1 이 UNK 로 기운다.
        {"record_id": "v1", "split": "val", "final": {"reason_label": "UNK"}},
        {"record_id": "x1", "split": "test", "final": {"reason_label": "DESIGN"}},
    ]
    records_path, labels_path = tmp_path / "records.jsonl", tmp_path / "labels.jsonl"
    records_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    labels_path.write_text("\n".join(json.dumps(r) for r in labels), encoding="utf-8")
    return ["--records", str(records_path), "--labels", str(labels_path)]


def test_cli_split_trains_on_train_and_predicts_only_that_split(tmp_path, capsys):
    """--split val 은 학습 split 으로만 학습하고 검증 split 만 낸다 (#86)."""
    out_path = tmp_path / "out.jsonl"

    code = clf.main(split_files(tmp_path) + ["--split", "val", "--out", str(out_path)])

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert code == 0
    assert [(row["record_id"], row["predicted_label"]) for row in rows] == [("v1", "DEAD")]
    assert "학습 레코드와 이어진 2건" in capsys.readouterr().out


def test_cli_refuses_test_without_the_final_flag(tmp_path, capsys):
    """test 는 최종 1회만 - 실수로 돌리면 그 자체가 test 를 본 것이 된다 (게이트 2 사전 등록)."""
    out_path = tmp_path / "out.jsonl"

    refused = clf.main(split_files(tmp_path) + ["--split", "test", "--out", str(out_path)])

    assert refused == 2
    assert not out_path.exists()
    assert "--final-test" in capsys.readouterr().err
    final = clf.main(split_files(tmp_path) + ["--split", "test", "--final-test"])
    assert final == 0
    assert "학습 레코드와 이어진 3건" in capsys.readouterr().out  # train + val 로 학습
