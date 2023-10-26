"""
Quick tests to check the version_info debugging for users works as intended.
"""
import platform

from xibs.version import VERSION, version_info


def test_version_info(capsys):
    # Check at least that we report the right version and platform
    info_str: str = version_info()
    assert f"XIBS version: {VERSION}" in info_str
    assert platform.platform() in info_str

    # Same thing, but now with when printed to stdout
    print(info_str)
    captured = capsys.readouterr()
    assert f"XIBS version: {VERSION}" in captured.out
    assert platform.platform() in captured.out
