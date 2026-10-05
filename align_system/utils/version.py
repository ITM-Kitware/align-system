def get_version() -> str:
    """Return the current align-system version string.

    In a development git checkout, uses setuptools-scm to produce an exact
    git-describe string (e.g. ``0.5.10.dev3+g1234567``). Falls back to the
    installed package metadata when the git repo is unavailable (e.g. in a
    packaged or containerized environment).
    """
    try:
        from setuptools_scm import get_version
        return get_version(root="../..", relative_to=__file__)
    except Exception:
        from importlib.metadata import version
        return version("align-system")
