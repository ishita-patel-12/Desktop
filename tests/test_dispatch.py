"""
Regression tests for executor/dispatch.py's fallback behavior.
Run with: pytest tests/test_dispatch.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from intent.schema import Intent, IntentType
from executor.dispatch import dispatch


def test_open_app_falls_back_to_website_when_app_unrecognized():
    """'open youtube' - 'youtube' isn't an installed app in config/apps.yaml,
    so this used to just fail outright. It should now open youtube.com in
    the default browser instead, without youtube specifically being
    hardcoded anywhere (the fallback is name -> name.com generically)."""
    intent = Intent(type=IntentType.OPEN_APP, app_name="youtube", raw_text="open youtube")

    with patch("executor.browser.webbrowser.open", return_value=True) as mock_open:
        result = dispatch(intent)

    assert result.success
    assert "youtube.com" in mock_open.call_args[0][0]


def test_open_app_falls_back_for_arbitrary_unrecognized_name():
    """Same fallback for a name never seen anywhere in this codebase or its
    tests, proving this is a generic heuristic and not a hidden lookup
    table of specific site names."""
    intent = Intent(type=IntentType.OPEN_APP, app_name="somebrandnewsite", raw_text="open somebrandnewsite")

    with patch("executor.browser.webbrowser.open", return_value=True) as mock_open:
        result = dispatch(intent)

    assert result.success
    assert "somebrandnewsite.com" in mock_open.call_args[0][0]


def test_open_app_still_prefers_known_installed_app():
    """Recognized apps must still launch as an app, not get diverted to
    the website fallback."""
    intent = Intent(type=IntentType.OPEN_APP, app_name="chrome", raw_text="open chrome")

    with patch("executor.apps.subprocess.Popen") as mock_popen:
        result = dispatch(intent)

    assert result.success
    mock_popen.assert_called_once()