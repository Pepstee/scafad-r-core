# tests/test_data_generator.py
from typing import Dict, List
import random

def generate_synthetic_workloads(n_functions: int = 5, duration: int = 60) -> List[Dict]:
    random.seed(0)
    return [{"function": f"f{i}", "payload": {"i": i}} for i in range(n_functions)]

def create_adversarial_test_cases(attack_types: List[str]) -> List[Dict]:
    return [{"attack": t, "payload": {"noise": 0.1}} for t in attack_types]

def generate_economic_abuse_scenarios() -> List[Dict]:
    return [{"pattern": "burst"}, {"pattern": "plateau"}]

def test_generators_smoke():
    assert len(generate_synthetic_workloads()) > 0
    assert len(create_adversarial_test_cases(["evasion"])) > 0
    assert len(generate_economic_abuse_scenarios()) > 0
