from tests.helper_for_testing import run_test
import pytest

from pulsedjax.streaking import (GeneralizedProjection, COPRA, DifferentialEvolution, Evosax, AutoDiff)


algorithms_list = [GeneralizedProjection, COPRA, 
                   DifferentialEvolution, Evosax, 
                   AutoDiff
                   ]

@pytest.mark.parametrize("algorithm", algorithms_list)
def test_streaking(algorithm):
    for i in range(5):
        run_test(i, "streaking", algorithm, real_fields=False)

