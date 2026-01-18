from tests.helper_for_testing import run_test
import pytest

from pulsedjax.twodsi import (DirectReconstruction, GeneralizedProjection, PtychographicIterativeEngine, COPRA, 
                           DifferentialEvolution, Evosax, LSF, AutoDiff)

algorithms_list = [DirectReconstruction, GeneralizedProjection, PtychographicIterativeEngine, COPRA, 
                   DifferentialEvolution, Evosax, LSF, AutoDiff]

@pytest.mark.parametrize("algorithm", algorithms_list)
def test_2dsi(algorithm):
    for i in range(5):
        test = run_test(i, "twodsi", algorithm, real_fields=False)








from pulsedjax.real_fields import twodsi

algorithms_list = [twodsi.PtychographicIterativeEngine, twodsi.DifferentialEvolution, twodsi.Evosax, twodsi.LSF, twodsi.AutoDiff]

@pytest.mark.parametrize("algorithm", algorithms_list)
def test_2dsi_real_fields(algorithm):
    for i in range(5):
        test = run_test(i, "twodsi", algorithm, real_fields=True)
