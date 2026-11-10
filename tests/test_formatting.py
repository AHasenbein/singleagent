from ui.formatting import (escape_dollars, evidence_rows, for_display, link_citations,
                           source_rows, without_sources)

RECORDS = [
    {"claim": "a", "source": "S1", "url": "https://one.example/a"},
    {"claim": "b", "source": "S2", "url": "https://two.example/b"},
    {"claim": "c", "source": "S1", "url": "https://one.example/a"},
    {"claim": "d", "source": "S3", "url": "not a url"},
]


def test_single_and_grouped_citations_link_to_sources():
    out = link_citations("X [E1]. Y [E2, E3]; Z [E2; E1].", RECORDS)
    assert "[[E1]](<https://one.example/a>)" in out
    assert "[[E2]](<https://two.example/b>) [[E3]](<https://one.example/a>)" in out


def test_unknown_or_unlinkable_labels_stay_plain():
    assert link_citations("[E9] and [E4]", RECORDS) == "[E9] and [E4]"


def test_non_citation_brackets_untouched():
    assert link_citations("[see E1] and [note]", RECORDS) == "[see E1] and [note]"


def test_dollars_escaped_once():
    assert escape_dollars("from $46B to $54B") == r"from \$46B to \$54B"
    assert escape_dollars(r"already \$5") == r"already \$5"
    assert for_display("$5 [E1]", RECORDS).startswith(r"\$5 [[E1]]")


def test_without_sources_strips_reference_section():
    md = "## Executive Summary\n\nText\n\n## Sources / References\n\n- [E1] S1"
    assert without_sources(md) == "## Executive Summary\n\nText"
    assert without_sources("no refs") == "no refs"


def test_rows_use_positional_labels_and_group_sources():
    assert [r["id"] for r in evidence_rows(RECORDS)] == ["E1", "E2", "E3", "E4"]
    top = source_rows(RECORDS)[0]
    assert top["url"] == "https://one.example/a" and top["records"] == 2 and top["labels"] == "E1, E3"
