import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "make_osr_user_tasks.py"
SPEC = importlib.util.spec_from_file_location(
    "make_osr_user_tasks_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
osr_task_gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(osr_task_gen)


def test_select_leaves_right_to_left_is_unique_and_deterministic() -> None:
    leaves = [
        {"page_index": 10, "breadcrumb": "/0"},
        {"page_index": 11, "breadcrumb": "/1"},
        {"page_index": 12, "breadcrumb": "/1/0"},
        {"page_index": 13, "breadcrumb": "/1/1"},
    ]

    selected = osr_task_gen._select_leaves_right_to_left(leaves, count=2)

    assert [leaf["page_index"] for leaf in selected] == [13, 12]


def test_navigation_hint_uses_requested_remaining_depth() -> None:
    hint = osr_task_gen._build_navigation_hint(
        hint_breadcrumb="/1/1/1",
        remaining_depth=2,
    )

    assert (
        "Breadcrumb convention: `/a/b/...` indexes only forward child-navigation "
        "buttons at each step; 0=first child, 1=second child."
        in hint
    )
    assert (
        'Do not count the "Back to parent" button when decoding breadcrumb digits, '
        "although you may still use it to backtrack during exploration."
        in hint
    )
    assert (
        "Example: if a page shows [Back to parent, Child 0, Child 1], "
        "breadcrumb digit `0` means click Child 0, and digit `1` means click Child 1."
        in hint
    )
    assert (
        "then continue searching nearby pages to find the relevant webpage and "
        "answer the question below."
        in hint
    )
    assert "breadcrumb `/1/1/1`" in hint
    assert "2 forward child-navigation steps" not in hint


def test_query_prompt_requires_exact_title() -> None:
    prompt = osr_task_gen._build_query_prompt(
        title_text="Meal Plan Confirmation & Final Dietary Review",
        body_text=(
            "This page allows you to review and confirm your finalized custom meal plan. "
            "You can make final adjustments or request a final consultation if needed."
        ),
    )

    assert 'exact full page title: "Meal Plan Confirmation & Final Dietary Review"' in prompt
    assert 'Find the page titled "Meal Plan Confirmation & Final Dietary Review"' in prompt


def test_fallback_query_explicitly_includes_title() -> None:
    query = osr_task_gen._fallback_query(
        title_text="Meal Plan Confirmation & Final Dietary Review",
        body_text=(
            "This page allows you to review and confirm your finalized custom meal plan. "
            "You can make final adjustments or request a final consultation if needed."
        ),
    )

    assert 'Find the page titled "Meal Plan Confirmation & Final Dietary Review"' in query
    assert "what is the next sentence after" in query


def test_generate_query_uses_llm_output() -> None:
    class _Message:
        content = (
            'Find the page titled "Meal Plan Confirmation & Final Dietary Review". '
            'On that page, what exact phrase immediately follows "You can make final adjustments or request a final consultation if needed."?'
        )

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    calls: list[dict] = []
    original_chat = osr_task_gen.llm_chat

    def _fake_chat(**kwargs):
        calls.append(kwargs)
        return _Response()

    osr_task_gen.llm_chat = _fake_chat
    try:
        query, generation = osr_task_gen._generate_query(
            title_text="Meal Plan Confirmation & Final Dietary Review",
            body_text=(
                "This page allows you to review and confirm your finalized custom meal plan. "
                "You can make final adjustments or request a final consultation if needed."
            ),
            model="gpt-4o-mini",
            cache_ttl=86400,
            max_tokens=128,
            temperature=0.0,
            allow_fallback=False,
        )
    finally:
        osr_task_gen.llm_chat = original_chat

    assert calls
    assert calls[0]["namespace"] == "stage4_osr_user_query"
    assert 'Find the page titled "Meal Plan Confirmation & Final Dietary Review"' in query
    assert generation["query"] == query
    assert "prompt" in generation
