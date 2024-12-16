from ev_fleet_benchmark import __version__


def test_version_is_available_from_source_checkout() -> None:
    assert __version__
