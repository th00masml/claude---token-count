from t2sbench.strategies.basic import S0, S1, S2
from t2sbench.strategies.consistency import S5
from t2sbench.strategies.explore import S3
from t2sbench.strategies.repair import S4

STRATEGIES = {cls.name: cls for cls in (S0, S1, S2, S3, S4, S5)}


def get_strategy(name: str):
    return STRATEGIES[name]()
