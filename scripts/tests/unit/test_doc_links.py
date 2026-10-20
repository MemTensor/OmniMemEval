from check_doc_links import missing_links


def test_existing_relative_links_and_external_urls(tmp_path):
    (tmp_path / "target file.md").write_text("# Target\n", encoding="utf-8")
    document = tmp_path / "README.md"
    document.write_text(
        '[file](target%20file.md#target)\n'
        '[title](target%20file.md "A title")\n'
        '[space](<target file.md>)\n'
        '[web](https://example.com/missing)\n'
        '[heading](#local-heading)\n',
        encoding="utf-8",
    )
    assert list(missing_links(document)) == []


def test_missing_link_reports_document_line(tmp_path):
    document = tmp_path / "README.md"
    document.write_text("# Guide\n\n[missing](missing.md#section)\n", encoding="utf-8")
    assert list(missing_links(document)) == [(3, "missing.md#section")]


def test_fenced_examples_do_not_count_as_links(tmp_path):
    document = tmp_path / "README.md"
    document.write_text(
        "````markdown\n[example](not-a-file.md)\n```\n````\n"
        "~~~markdown\n[example](also-not-a-file.md)\n~~~\n",
        encoding="utf-8",
    )
    assert list(missing_links(document)) == []
