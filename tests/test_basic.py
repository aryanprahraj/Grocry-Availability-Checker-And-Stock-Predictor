"""Minimal test to verify pytest is working"""


def test_always_passes():
    """This test should always pass"""
    assert 1 + 1 == 2


def test_python_version():
    """Test Python version"""
    import sys
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 9
