from tests.helper_for_testing import run_test
import pytest

from pulsedjax.tdp import (LSGPA, CPCGPA, GeneralizedProjection, PtychographicIterativeEngine, COPRA, 
                           DifferentialEvolution, Evosax, LSF, AutoDiff)

algorithms_list = [LSGPA, CPCGPA, GeneralizedProjection, PtychographicIterativeEngine, COPRA, 
                   DifferentialEvolution, Evosax, LSF, AutoDiff]


@pytest.mark.parametrize("algorithm", algorithms_list)
def test_tdp(algorithm):
    for i in range(5):
        test = run_test(i, "tdp", algorithm, real_fields=False)






from pulsedjax.real_fields import tdp 

algorithms_list = [tdp.PtychographicIterativeEngine, tdp.DifferentialEvolution, tdp.Evosax, tdp.LSF, tdp.AutoDiff]

@pytest.mark.parametrize("algorithm", algorithms_list)
def test_tdp_real_fields(algorithm):
    for i in range(5):
        test = run_test(i, "tdp", algorithm, real_fields=True)