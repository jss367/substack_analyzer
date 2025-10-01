from substack_analyzer.types import AdSpendSchedule, SimulationInputs


def test_adspend_constant_and_two_stage():
    const = AdSpendSchedule.constant(500.0)
    assert const.get_spend_for_month(0) == 500.0
    assert const.get_spend_for_month(36) == 500.0

    two = AdSpendSchedule.two_stage(3000.0, 1000.0)
    assert two.get_spend_for_month(0) == 3000.0
    assert two.get_spend_for_month(23) == 3000.0
    assert two.get_spend_for_month(24) == 1000.0


def test_simulation_inputs_defaults():
    s = SimulationInputs()
    assert s.horizon_months > 0
    assert 0.0 <= s.substack_fee_pct <= 1.0
