from tests.helper_for_testing import run_test
import pytest

from pulsedjax.chirp_scan import (Basic, GeneralizedProjection, PtychographicIterativeEngine, COPRA, 
                                  DifferentialEvolution, Evosax, LSF, AutoDiff)

algorithms_list = [Basic, GeneralizedProjection, PtychographicIterativeEngine, COPRA, 
                   DifferentialEvolution, Evosax, LSF, AutoDiff]


@pytest.mark.parametrize("algorithm", algorithms_list)
def test_chirp_scan(algorithm):
    for i in range(5):
        test = run_test(i, "chirp_scan", algorithm, real_fields=False)






from pulsedjax.real_fields import chirp_scan 

algorithms_list = [chirp_scan.PtychographicIterativeEngine, 
                   chirp_scan.DifferentialEvolution, chirp_scan.Evosax, chirp_scan.LSF, chirp_scan.AutoDiff]


@pytest.mark.parametrize("algorithm", algorithms_list)
def test_chirp_scan_real_fields(algorithm):
    for i in range(5):
        test = run_test(i, "chirp_scan", algorithm, real_fields=True)