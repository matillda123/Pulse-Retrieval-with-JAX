from tests.helper_for_testing import run_test
import pytest

from pulsedjax.frog import (Vanilla, LSGPA, CPCGPA, GeneralizedProjection, PtychographicIterativeEngine, 
                            COPRA, DifferentialEvolution, Evosax, LSF, AutoDiff)


algorithms_list = [Vanilla, LSGPA, CPCGPA, GeneralizedProjection, PtychographicIterativeEngine, COPRA, 
                   DifferentialEvolution, Evosax, LSF, AutoDiff]

@pytest.mark.parametrize("algorithm", algorithms_list)
def test_frog(algorithm):
    for i in range(5):
        run_test(i, "frog", algorithm, real_fields=False)





from pulsedjax.real_fields import frog 

algorithms_list = [frog.PtychographicIterativeEngine, frog.DifferentialEvolution, frog.Evosax, frog.LSF, frog.AutoDiff]

@pytest.mark.parametrize("algorithm", algorithms_list)
def test_frog_real_fields(algorithm):
    for i in range(5):
        run_test(i, "frog", algorithm, real_fields=True)



