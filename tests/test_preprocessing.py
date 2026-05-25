#!/usr/bin/env python3
"""Unit tests for text preprocessing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from speeker.preprocessing import preprocess_for_tts


def test_arrows():
    assert "to" in preprocess_for_tts("A → B")
    assert "from" in preprocess_for_tts("A ← B")
    assert "between" in preprocess_for_tts("A ↔ B")
    assert "implies" in preprocess_for_tts("A ⇒ B")


def test_math_symbols():
    assert "approximately" in preprocess_for_tts("≈ 95")
    assert "not equal" in preprocess_for_tts("x ≠ y")
    assert "less than or equal" in preprocess_for_tts("x ≤ 10")
    assert "greater than or equal" in preprocess_for_tts("x ≥ 5")
    assert "degrees" in preprocess_for_tts("72°")


def test_percentages():
    result = preprocess_for_tts("95%")
    assert "95" in result
    assert "percent" in result


def test_abbreviations():
    assert "for example" in preprocess_for_tts("e.g. this")
    assert "that is" in preprocess_for_tts("i.e. this")
    assert "etcetera" in preprocess_for_tts("etc.")
    assert "versus" in preprocess_for_tts("A vs. B")
    assert "versus" in preprocess_for_tts("A vs B")


def test_file_paths():
    result = preprocess_for_tts("~/projects")
    assert "home" in result
    assert "slash" in result

    result = preprocess_for_tts("./script")
    assert "dot" in result
    assert "slash" in result

    result = preprocess_for_tts("../parent")
    assert "dot dot slash" in result


def test_file_extensions():
    result = preprocess_for_tts("script.sh")
    assert "dot sh" in result

    result = preprocess_for_tts("file.py")
    assert "dot py" in result


def test_path_slashes():
    result = preprocess_for_tts("path/to/file")
    assert "slash" in result
    assert "/" not in result


def test_version_numbers():
    result = preprocess_for_tts("version 3.11")
    assert "point" in result
    assert "." not in result

    result = preprocess_for_tts("version 1.2.3")
    assert result.count("point") == 2


def test_programming_operators():
    assert "equals" in preprocess_for_tts("a == b")
    assert "not equals" in preprocess_for_tts("a != b")
    assert " and " in preprocess_for_tts("a && b")
    assert " or " in preprocess_for_tts("a || b")


def test_mixed_content():
    result = preprocess_for_tts("Check ~/config → saved. Value ≈ 95%, i.e. done.")
    assert "home" in result
    assert "slash" in result
    assert "to" in result
    assert "approximately" in result
    assert "percent" in result
    assert "that is" in result


def test_tool_name_pronunciation():
    # python's "uv" must not be read as the word "oove"
    assert "you vee" in preprocess_for_tts("run uv sync")
    assert "you vee x" in preprocess_for_tts("uvx ruff")
    # case-insensitive
    assert "you vee" in preprocess_for_tts("UV light")


def test_todo_pronunciation():
    # "todo" must read as "to do", not "TOTO"
    assert "to do" in preprocess_for_tts("update the todo list")
    assert "to dos" in preprocess_for_tts("clear all todos")
    # not mangled inside other words
    assert preprocess_for_tts("mikado") == "mikado"


def test_tool_name_word_boundary():
    # Substrings inside real words must be left alone
    result = preprocess_for_tts("the Louvre museum")
    assert "Louvre" in result
    assert "you vee" not in result


def test_known_acronyms_spelled():
    # Built-in acronyms (PHI, PII, SSN, DOB) are spelled letter-by-letter
    assert "P H I" in preprocess_for_tts("PHI was exposed")
    assert "S S N" in preprocess_for_tts("the SSN field")


def test_unknown_caps_left_alone():
    # Non-acronym ALL-CAPS words aren't spelled or title-cased
    result = preprocess_for_tts("DEPLOYED to production")
    assert "DEPLOYED" in result


def test_custom_acronyms_param():
    # Caller-supplied acronyms (player passes the configured set) are spelled
    result = preprocess_for_tts("the API is down", acronyms={"API"})
    assert "A P I" in result
    # ... but not when it isn't a known acronym
    assert "A P I" not in preprocess_for_tts("the API is down", acronyms=set())


def test_cleanup():
    # Multiple spaces should be collapsed
    result = preprocess_for_tts("a  →  b")
    assert "  " not in result

    # Space before punctuation should be removed
    result = preprocess_for_tts("hello .")
    assert " ." not in result


def test_empty_input():
    assert preprocess_for_tts("") == ""
    assert preprocess_for_tts(None) is None


if __name__ == "__main__":
    import traceback

    tests = [
        test_arrows,
        test_math_symbols,
        test_percentages,
        test_abbreviations,
        test_file_paths,
        test_file_extensions,
        test_path_slashes,
        test_version_numbers,
        test_programming_operators,
        test_mixed_content,
        test_tool_name_pronunciation,
        test_todo_pronunciation,
        test_tool_name_word_boundary,
        test_known_acronyms_spelled,
        test_unknown_caps_left_alone,
        test_custom_acronyms_param,
        test_cleanup,
        test_empty_input,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
