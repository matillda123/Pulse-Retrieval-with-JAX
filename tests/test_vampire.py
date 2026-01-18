from tests.helper_for_testing import run_test
import pytest

from pulsedjax.vampire import (LSGPA, CPCGPA, GeneralizedProjection, PtychographicIterativeEngine, COPRA, 
                           DifferentialEvolution, Evosax, LSF, AutoDiff)

algorithms_list = [LSGPA, CPCGPA, GeneralizedProjection, PtychographicIterativeEngine, COPRA, 
                   DifferentialEvolution, Evosax, LSF, AutoDiff]


@pytest.mark.parametrize("algorithm", algorithms_list)
def test_vampire(algorithm):
    for i in range(5):
        test = run_test(i, "vampire", algorithm, real_fields=False)









from pulsedjax.real_fields import vampire

algorithms_list = [vampire.PtychographicIterativeEngine, vampire.DifferentialEvolution, 
                   vampire.Evosax, vampire.LSF, vampire.AutoDiff]

@pytest.mark.parametrize("algorithm", algorithms_list)
def test_vampire_real_fields(algorithm):
    for i in range(5):
        test = run_test(i, "vampire", algorithm, real_fields=True)
