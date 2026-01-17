from pulsedjax.frog import (Vanilla, LSGPA, CPCGPA, GeneralizedProjection, PtychographicIterativeEngine, 
                            COPRA, DifferentialEvolution, Evosax, LSF, AutoDiff)

from tests.helper_for_testing import run_test
import pytest



algorithms_list = [Vanilla, LSGPA, CPCGPA, GeneralizedProjection, PtychographicIterativeEngine, COPRA, 
                   DifferentialEvolution, Evosax, LSF, AutoDiff]

@pytest.mark.parametrize("algorithm", algorithms_list)
def test_frog(algorithm):
    for i in range(5):
        run_test(i, "frog", algorithm, real_fields=False)



