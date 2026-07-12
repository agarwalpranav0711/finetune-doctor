from unittest.mock import patch, PropertyMock
from finetune_doctor.watch import _get_safe_symbol, console, _render_diagnosis
from finetune_doctor.capture import CapturedContext
from finetune_doctor.signatures.loader import Signature


def test_get_safe_symbol_utf8():
    # If encoding is utf-8, it should return the emoji
    with patch("rich.console.Console.encoding", new_callable=PropertyMock) as mock_encoding:
        mock_encoding.return_value = "utf-8"
        assert _get_safe_symbol("🩺", "[finetune-doctor]") == "🩺"
        assert _get_safe_symbol("⚠", "WARNING:") == "⚠"


def test_get_safe_symbol_ascii():
    # If encoding is ascii, it should return the fallback
    with patch("rich.console.Console.encoding", new_callable=PropertyMock) as mock_encoding:
        mock_encoding.return_value = "ascii"
        assert _get_safe_symbol("🩺", "[finetune-doctor]") == "[finetune-doctor]"
        assert _get_safe_symbol("⚠", "WARNING:") == "WARNING:"


def test_render_diagnosis_does_not_contain_escapes():
    sig = Signature(
        id="test_oom",
        name="Out of memory",
        specificity=50,
        match={"exception_type": "RuntimeError"},
        explanation="OOM explanation",
        fix="Fix it",
    )
    ctx = CapturedContext(
        exception_type="RuntimeError",
        exception_message="Out of memory",
        traceback_text="Traceback",
    )

    with patch.object(console, "print") as mock_print:
        _render_diagnosis(sig, ctx)
        assert mock_print.called

        # Check that none of the printed args contain literal escaped unicode text
        for call in mock_print.call_args_list:
            for arg in call[0]:
                if isinstance(arg, str):
                    assert "\\U" not in arg
                    assert "\\u" not in arg
