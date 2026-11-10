import pytest

from research_agent.llm import parse_json_object


@pytest.mark.parametrize("text", [
    '{"status": "PASS", "issues": []}',
    '  {"status": "PASS", "issues": []}  \n',
    '```json\n{"status": "PASS", "issues": []}\n```',
    '```\n{"status": "PASS", "issues": []}\n```\n',
    'Here is the JSON:\n{"status": "PASS", "issues": []}',
    'Here you go:\n```json\n{"status": "PASS", "issues": []}\n```',
])
def test_accepts_single_object(text):
    assert parse_json_object(text) == {"status": "PASS", "issues": []}


@pytest.mark.parametrize("text", [
    '{"status": "PASS"} {"status": "FAIL"}',
    '{"status": "PASS"}\n{"status": "FAIL"}',
    '{"status": "PASS"} and that is my verdict',
    '```json\n{"status": "PASS"}\n```\nextra',
    '',
    '   ',
    'no json here',
    '{"status": "PASS", "issues": [',
    '[{"status": "PASS"}]',
    '{"a": 1',
])
def test_rejects_ambiguous_or_broken_replies(text):
    with pytest.raises(ValueError):
        parse_json_object(text)


def test_none_is_rejected():
    with pytest.raises(ValueError):
        parse_json_object(None)
