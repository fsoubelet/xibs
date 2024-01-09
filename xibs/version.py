VERSION = "0.4.0"


def version_info() -> str:
    """
    .. versionadded:: 0.2.0

    Debug convenience function to give version, platform and runtime information.
    """
    import pathlib
    import platform
    import sys

    info = {
        "XIBS version": VERSION,
        "Install path": pathlib.Path(__file__).resolve().parent,
        "Python version": f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}",
        "Python implementation": sys.version,
        "Platform": platform.platform(),
    }
    return "\n".join("{:>24} {}".format(k + ":", str(v).replace("\n", " ")) for k, v in info.items())
