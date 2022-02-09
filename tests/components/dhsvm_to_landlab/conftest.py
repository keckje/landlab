import pytest

import numpy as np
from landlab import RasterModelGrid


@pytest.fixture
def example_raster_model_grid():
    grid = RasterModelGrid((5, 4), xy_spacing=(0.2, 0.2))
    gridnum = grid.number_of_nodes
    np.random.seed(seed=7)

      
    return (grid)

