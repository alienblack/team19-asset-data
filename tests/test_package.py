"""Minimal check that the starter package remains importable."""

from team19_asset_data import __doc__


def test_package_has_description() -> None:
    assert __doc__
