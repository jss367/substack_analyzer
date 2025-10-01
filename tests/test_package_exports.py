import substack_analyzer as sa


def test_package_exports():
    # Ensure the package exposes expected symbols
    assert hasattr(sa, "simulate_growth")
    assert hasattr(sa, "SimulationInputs")
    assert hasattr(sa, "AdSpendSchedule")
