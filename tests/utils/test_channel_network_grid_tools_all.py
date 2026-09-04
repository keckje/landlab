import numpy as np
import pytest
from numpy.testing import assert_array_equal

import landlab.utils.channel_network_grid_tools_all as gt
from landlab import RasterModelGrid
from landlab.components import DepressionFinderAndRouter
from landlab.components import FlowAccumulator
from landlab.grid.create_network import AtMostNodes
from landlab.grid.create_network import network_grid_from_raster


def check_vals(vals, vals_e):
    np.testing.assert_allclose(vals, vals_e, atol=1e-4)


@pytest.fixture
def grid():
    """small watershed used for tests, drains to the south"""
    mg = RasterModelGrid((8, 7), 10)
    mg.at_node["topographic__elevation"] = np.array(
        [
            [5, 3, 2, 1, 2, 3, 5],
            [7, 4, 1.5, 3, 3, 4, 7],
            [7.5, 5, 4, 2, 4, 5, 7.5],
            [7.5, 5, 3, 5, 6, 6.5, 9],
            [9, 5.5, 4, 4.1, 5.5, 8, 10],
            [10.5, 6, 8, 7, 7, 7, 11],
            [12, 7, 9, 8, 9, 10, 12],
            [14, 12, 11.5, 11, 11.5, 12, 14],
        ]
    ).reshape(-1)
    mg.at_node["id"] = mg.nodes
    # add flow direction and drainage area and correct for any pits
    fr = FlowAccumulator(mg, "topographic__elevation", flow_director="D8")  # add
    fr.run_one_step()
    df_4 = DepressionFinderAndRouter(mg)
    df_4.map_depressions()
    return mg


@pytest.fixture
def grid_up():
    """small watershed used for tests, flipped upside down, drains to the north"""
    mg_up = RasterModelGrid((8, 7), 10)
    mg_up.at_node["topographic__elevation"] = np.flipud(
        np.array(
            [
                5,
                3,
                2,
                1,
                2,
                3,
                5,
                7,
                4,
                1.5,
                3,
                3,
                4,
                7,
                7.5,
                5,
                4,
                2,
                4,
                5,
                7.5,
                7.5,
                5,
                3,
                5,
                6,
                6.5,
                9,
                9,
                5.5,
                4,
                4.1,
                5.5,
                8,
                10,
                10.5,
                6,
                8,
                7,
                7,
                7,
                11,
                12,
                7,
                9,
                8,
                9,
                10,
                12,
                14,
                12,
                11.5,
                11,
                11.5,
                12,
                14,
            ]
        )
    )
    mg_up.at_node["id"] = mg_up.nodes
    # add flow direction and drainage area and correct for any pits
    fr = FlowAccumulator(mg_up, "topographic__elevation", flow_director="D8")  # add
    fr.run_one_step()
    df_4 = DepressionFinderAndRouter(mg_up)
    df_4.map_depressions()
    return mg_up


@pytest.fixture
def nmgrid_f(grid):
    """fine-scale network model grid representation of watershed"""
    nmg_f = network_grid_from_raster(
        grid,
        reducer=AtMostNodes(count=4),
        minimum_channel_threshold=200,  # upstream drainage area to truncate network, in m^2
        include=["drainage_area", "topographic__elevation"],
    )
    nmg_f.at_link["drainage_area"] = nmg_f.map_mean_of_link_nodes_to_link(
        "drainage_area"
    )
    return nmg_f


@pytest.fixture
def nmgrid_f_up(grid_up):
    """fine-scale network model grid representation of watershed, flipped
    upside down"""
    nmg_f_up = network_grid_from_raster(
        grid_up,
        reducer=AtMostNodes(count=4),
        minimum_channel_threshold=200,  # upstream drainage area to truncate network, in m^2
        include=["drainage_area", "topographic__elevation"],
    )
    nmg_f_up.at_link["drainage_area"] = nmg_f_up.map_mean_of_link_nodes_to_link(
        "drainage_area"
    )
    return nmg_f_up


@pytest.fixture
def nmgrid_c(grid):
    """coarse-scale network model grid representation of watershed"""
    nmg_c = network_grid_from_raster(
        grid,
        reducer=AtMostNodes(count=2),
        minimum_channel_threshold=300,
        include=["drainage_area", "topographic__elevation"],
    )
    nmg_c.at_link["drainage_area"] = nmg_c.map_mean_of_link_nodes_to_link(
        "drainage_area"
    )
    return nmg_c


@pytest.fixture
def nmgrid_c_up(grid_up):
    """coarse-scale network model grid representation of watershed, flipped
    upside down"""
    nmg_c_up = network_grid_from_raster(
        grid_up,
        reducer=AtMostNodes(count=2),
        minimum_channel_threshold=300,
        include=["drainage_area", "topographic__elevation"],
    )
    nmg_c_up.at_link["drainage_area"] = nmg_c_up.map_mean_of_link_nodes_to_link(
        "drainage_area"
    )
    return nmg_c_up


@pytest.fixture
def nmgrid_s(grid):
    """single-channel-reach network model grid representation of watershed"""
    nmg_s = network_grid_from_raster(
        grid,
        reducer=AtMostNodes(count=2),
        minimum_channel_threshold=1000,
        include=["drainage_area", "topographic__elevation"],
    )
    nmg_s.at_link["drainage_area"] = nmg_s.map_mean_of_link_nodes_to_link(
        "drainage_area"
    )
    return nmg_s


@pytest.fixture
def nmgrid_s_up(grid_up):
    """single-channel-reach network model grid representation of watershed, flipped
    upside down"""
    nmg_s_up = network_grid_from_raster(
        grid_up,
        reducer=AtMostNodes(count=2),
        minimum_channel_threshold=1000,
        include=["drainage_area", "topographic__elevation"],
    )
    nmg_s_up.at_link["drainage_area"] = nmg_s_up.map_mean_of_link_nodes_to_link(
        "drainage_area"
    )
    return nmg_s_up


@pytest.mark.parametrize(
    "array, choose, expected",
    (
        ([0, 0, 1, 1, 1, 2, 2], "last", [0, 1, 0, 0, 1, 0, 1]),
        ([2, 2, 2, 0, 0, 1, 1, 1], "last", [0, 0, 1, 0, 1, 0, 0, 1]),
        ([0, 0, 1, 1, 1, 2, 2], "first", [1, 0, 1, 0, 0, 1, 0]),
        ([2, 2, 2, 0, 0, 1, 1, 1], "first", [1, 0, 0, 1, 0, 1, 0, 0]),
        ([2, 2, 2], "first", [1, 0, 0]),
        ([2, 2, 2], "last", [0, 0, 1]),
    ),
)
def test_choose_from_repeated_normal(array, choose, expected):
    actual = gt.choose_from_repeated(array, choose=choose)
    assert_array_equal(actual, expected)
    assert actual.dtype == bool


@pytest.mark.parametrize("choose", ("first", "last"))
@pytest.mark.parametrize("size", (0, 1))
def test_choose_from_repeated_small_input(choose, size):
    actual = gt.choose_from_repeated(np.zeros(size, dtype=int), choose=choose)
    assert_array_equal(actual, np.ones(size, dtype=bool))
    assert actual.dtype == bool


@pytest.mark.parametrize("choose", ("foo", "FIRST", " last", "last ", ""))
def test_choose_from_repeated_bad_keyword(choose):
    with pytest.raises(ValueError, match="choose must be"):
        gt.choose_from_repeated([0, 0, 1, 1], choose=choose)


@pytest.mark.parametrize("choose", ("foo", "FIRST", " last", "last ", ""))
def test_choose_unique_bad_choose(choose):
    with pytest.raises(ValueError, match="choose must be"):
        gt.choose_unique([0, 1, 0], choose=choose)


@pytest.mark.parametrize(
    "order_by",
    (
        ([0, 1, 2, 3],),
        ([0, 1],),
        ([0, 1, 2], [0]),
    ),
)
def test_choose_unique_bad_order_by(order_by):
    with pytest.raises(ValueError, match="All `order_by` arrays must match"):
        gt.choose_unique([0, 1, 0], order_by=order_by)


def test_choose_unique_normal():
    actual = gt.choose_unique([10, 11, 10], choose="last")
    assert_array_equal(actual, [1, 2])

    actual = gt.choose_unique([10, 11, 10], choose="first")
    assert_array_equal(actual, [0, 1])


@pytest.mark.parametrize(
    "values, order_by, choose, expected",
    (
        ([10, 11, 10], ([5.0, 6.0, 4.0],), "last", [0, 1]),
        ([10, 11, 10], ([5.0, 6.0, 4.0],), "first", [1, 2]),
        ([10, 11, 10], ([4.0, 6.0, 5.0],), "last", [1, 2]),
        ([10, 11, 10], ([4.0, 6.0, 5.0],), "first", [0, 1]),
        ([10, 11, 10], ([4.0, 6.0, 5.0], [5.0, 6.0, 4.0]), "first", [1, 2]),
    ),
)
def test_choose_unique_with_order_by(values, order_by, choose, expected):
    actual = gt.choose_unique(values, order_by=order_by, choose=choose)
    assert_array_equal(actual, expected)


def test_dist_func():
    check_vals(gt._dist_func(0, 1, 0, 1), 1.414213)


class TestGetLinknodes:

    def test_get_link_nodes_coarse(self, nmgrid_c):
        """test using simple nmgrid"""
        link_nodes_c = gt.get_link_nodes(nmgrid_c)
        link_nodes_c_e = np.array([[0, 1], [1, 2], [1, 3]])
        check_vals(link_nodes_c, link_nodes_c_e)

    def test_get_link_nodes_coarse_up(self, nmgrid_c_up):
        """test using simple nmgrid"""
        link_nodes_c_up = gt.get_link_nodes(nmgrid_c_up)
        link_nodes_c_up_e = np.array([[2, 0], [2, 1], [3, 2]])
        check_vals(link_nodes_c_up, link_nodes_c_up_e)

    def test_get_link_nodes_fine(self, nmgrid_f):
        """test using a more complex nmgrid"""
        link_nodes_f = gt.get_link_nodes(nmgrid_f)
        link_nodes_f_e = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [4, 7]]
        )
        check_vals(link_nodes_f, link_nodes_f_e)

    def test_get_link_nodes_fine_up(self, nmgrid_f_up):
        """test using a more complex nmgrid"""
        link_nodes_f_up = gt.get_link_nodes(nmgrid_f_up)
        link_nodes_f_up_e = np.array(
            [[3, 0], [2, 1], [4, 2], [4, 3], [5, 4], [6, 5], [7, 6]]
        )
        check_vals(link_nodes_f_up, link_nodes_f_up_e)

    def test_get_link_nodes_single(self, nmgrid_s):
        """test using simple nmgrid"""
        link_nodes_s = gt.get_link_nodes(nmgrid_s)
        link_nodes_s_e = np.array([[0, 1]])
        check_vals(link_nodes_s, link_nodes_s_e)

    def test_get_link_nodes_single_up(self, nmgrid_s_up):
        """test using simple nmgrid"""
        link_nodes_s_up = gt.get_link_nodes(nmgrid_s_up)
        link_nodes_s_up_e = np.array([[1, 0]])
        check_vals(link_nodes_s_up, link_nodes_s_up_e)

    def test_get_link_nodes_slope_is_zero(self, nmgrid_c):
        """what happens if no slope?"""
        nmgrid_c.at_node["topographic__elevation"][
            1
        ] = 1  # set elevation of second node equal to first, so no slope
        link_nodes_c = gt.get_link_nodes(nmgrid_c)
        link_nodes_c_e = np.array(
            [
                [
                    -1,
                    -1,
                ],  # there are no downstream or upstream nodes, both listed as -1
                [1, 2],
                [1, 3],
            ]
        )
        check_vals(link_nodes_c, link_nodes_c_e)


class TestLinkToPointsAndDist:
    def test_link_to_points_and_dist_1(self):
        """from 0,0 to 1,1"""
        x0 = 0
        x1 = 1
        y0 = 0
        y1 = 1
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )

        v_e = (
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([0.0, 0.35355339, 0.70710678, 1.06066017, 1.41421356]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_2(self):
        """from 1,1 to 0,0"""
        x0 = 1
        x1 = 0
        y0 = 1
        y1 = 0
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([1.0, 0.75, 0.5, 0.25, 0.0]),
            np.array([1.0, 0.75, 0.5, 0.25, 0.0]),
            np.array([0.0, 0.35355339, 0.70710678, 1.06066017, 1.41421356]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_3(self):
        """from 0,0 to 1,-1"""
        x0 = 0
        x1 = 1
        y0 = 0
        y1 = -1
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([0.0, -0.25, -0.5, -0.75, -1.0]),
            np.array([0.0, 0.35355339, 0.70710678, 1.06066017, 1.41421356]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_4(self):
        """from 1,-1 to 0,0"""
        x0 = 1
        x1 = 0
        y0 = -1
        y1 = 0
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([1.0, 0.75, 0.5, 0.25, 0.0]),
            np.array([-1.0, -0.75, -0.5, -0.25, 0.0]),
            np.array([0.0, 0.35355339, 0.70710678, 1.06066017, 1.41421356]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_5(self):
        """from 0,0 to -1,-1"""
        x0 = 0
        x1 = -1
        y0 = 0
        y1 = -1
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([0.0, -0.25, -0.5, -0.75, -1.0]),
            np.array([0.0, -0.25, -0.5, -0.75, -1.0]),
            np.array([0.0, 0.35355339, 0.70710678, 1.06066017, 1.41421356]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_6(self):
        """from -1,-1 to 0,0"""
        x0 = -1
        x1 = 0
        y0 = -1
        y1 = 0
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([-1.0, -0.75, -0.5, -0.25, 0.0]),
            np.array([-1.0, -0.75, -0.5, -0.25, 0.0]),
            np.array([0.0, 0.35355339, 0.70710678, 1.06066017, 1.41421356]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_7(self):
        """from 0,0 to -1,1"""
        x0 = 0
        x1 = -1
        y0 = 0
        y1 = 1
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([0.0, -0.25, -0.5, -0.75, -1.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([0.0, 0.35355339, 0.70710678, 1.06066017, 1.41421356]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_8(self):
        """from -1,1 to 0,0"""
        x0 = -1
        x1 = 0
        y0 = 1
        y1 = 0
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([-1.0, -0.75, -0.5, -0.25, 0.0]),
            np.array([1.0, 0.75, 0.5, 0.25, 0.0]),
            np.array([0.0, 0.35355339, 0.70710678, 1.06066017, 1.41421356]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_9(self):
        """from 0,0 to 1,0"""
        x0 = 0
        x1 = 1
        y0 = 0
        y1 = 0
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_10(self):
        """from 1,0 to 0,0"""
        x0 = 1
        x1 = 0
        y0 = 0
        y1 = 0
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([1.0, 0.75, 0.5, 0.25, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_11(self):
        """from 0,0 to 0,-1"""
        x0 = 0
        x1 = 0
        y0 = 0
        y1 = -1
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, -0.25, -0.5, -0.75, -1.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_12(self):
        """from 0,-1 to 0,0"""
        x0 = 0
        x1 = 0
        y0 = -1
        y1 = 0
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-1.0, -0.75, -0.5, -0.25, 0.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_13(self):
        """from 0,0 to -1,0"""
        x0 = 0
        x1 = -1
        y0 = 0
        y1 = 0
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([0.0, -0.25, -0.5, -0.75, -1.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_14(self):
        """from -1,0 to 0,0"""
        x0 = -1
        x1 = 0
        y0 = 0
        y1 = 0
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([-1.0, -0.75, -0.5, -0.25, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_15(self):
        """from 0,0 to 0,1"""
        x0 = 0
        x1 = 0
        y0 = 0
        y1 = 1
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])

    def test_link_to_points_and_dist_16(self):
        """from 0,1 to 0,0"""
        x0 = 0
        x1 = 0
        y0 = 1
        y1 = 0
        x_, y_, dist_ = gt._link_to_points_and_dist(
            (x0, y0), (x1, y1), number_of_points=5
        )
        v_e = (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.75, 0.5, 0.25, 0.0]),
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        )
        check_vals(x_, v_e[0])
        check_vals(y_, v_e[1])
        check_vals(dist_, v_e[2])


class TestExtractChannelNodes:

    def test_extract_channel_nodes_1(self, grid):
        """large contributing area"""
        fcn = gt.extract_channel_nodes(grid, 1200)
        fcn_e = np.array([3, 9, 17, 23])
        check_vals(fcn, fcn_e)

    def test_extract_channel_nodes_2(self, grid):
        """small contributing area"""
        acn = gt.extract_channel_nodes(grid, 300)
        cn_e = np.array([3, 9, 17, 23, 30, 31, 32, 36])
        check_vals(acn, cn_e)

    def test_extract_channel_nodes_3(self, grid):
        """contributing area is zero"""
        acn = gt.extract_channel_nodes(grid, 0)
        cn_e = grid.nodes.flatten()  # should return all nodes
        check_vals(acn, cn_e)


class TestExtractTerraceNodes:

    def test_extract_terrace_nodes_1(self, grid):
        """terrace width = 1"""
        terrace_width = 1
        acn = np.array([3, 9, 17, 23, 30, 31, 32, 36])
        fcn = np.array([3, 9, 17, 23])
        tn = gt.extract_terrace_nodes(grid, terrace_width, acn, fcn)
        tn_e = np.array([1, 2, 4, 8, 10, 11, 15, 16, 18, 22, 24, 25, 29])
        check_vals(tn, tn_e)

    def test_extract_terrace_nodes_2(self, grid):
        """terrace width = 2"""
        terrace_width = 2
        acn = np.array([3, 9, 17, 23, 30, 31, 32, 36])
        fcn = np.array([3, 9, 17, 23])
        tn = gt.extract_terrace_nodes(grid, terrace_width, acn, fcn)
        tn_e = np.array(
            [
                0,
                1,
                2,
                4,
                5,
                7,
                8,
                10,
                11,
                12,
                14,
                15,
                16,
                18,
                19,
                21,
                22,
                24,
                25,
                26,
                28,
                29,
                33,
                35,
                37,
            ]
        )
        check_vals(tn, tn_e)

    def test_extract_terrace_nodes_3(self, grid):
        """terrace width = 0"""
        terrace_width = 0
        acn = np.array([3, 9, 17, 23, 30, 31, 32, 36])
        fcn = np.array([3, 9, 17, 23])
        with pytest.raises(ValueError) as exc_info:
            gt.extract_terrace_nodes(grid, terrace_width, acn, fcn)
        assert exc_info.match("terrace width must be 1 or greater")

    def test_extract_terrace_nodes_4(self, grid):
        """terrace width = 1.5 - should round to 2"""
        terrace_width = 1.5
        acn = np.array([3, 9, 17, 23, 30, 31, 32, 36])
        fcn = np.array([3, 9, 17, 23])
        tn = gt.extract_terrace_nodes(grid, terrace_width, acn, fcn)
        tn_e = np.array(
            [
                0,
                1,
                2,
                4,
                5,
                7,
                8,
                10,
                11,
                12,
                14,
                15,
                16,
                18,
                19,
                21,
                22,
                24,
                25,
                26,
                28,
                29,
                33,
                35,
                37,
            ]
        )
        check_vals(tn, tn_e)


class TestMinDistToNetwork:

    def test_min_dist_to_network_1(self, grid):
        """a normal test, only one closest node"""
        node_id = 47
        acn = np.array([3, 9, 17, 23, 30, 31, 32, 36])
        dist, mdn = gt.min_distance_to_network(grid, acn, node_id)
        dist_e = 22.36068
        mdn_e = 32
        check_vals(dist, dist_e)
        check_vals(mdn, mdn_e)

    def test_min_dist_to_network_2(self, grid):
        """two closest nodes, will pick first"""
        node_id = 2
        acn = np.array([3, 9, 17, 23, 30, 31, 32, 36])
        dist, mdn = gt.min_distance_to_network(grid, acn, node_id)
        dist_e = 10
        mdn_e = 3
        check_vals(dist, dist_e)
        check_vals(mdn, mdn_e)

    def test_min_dist_to_network_3(self, grid):
        """node coincides with a channel node"""
        node_id = 3
        acn = np.array([3, 9, 17, 23, 30, 31, 32, 36])
        dist, mdn = gt.min_distance_to_network(grid, acn, node_id)
        dist_e = 0
        mdn_e = 3
        check_vals(dist, dist_e)
        check_vals(mdn, mdn_e)


class TestMapNMGLinksToRMGCoincidentNodes:

    def test_map_nmg_links_to_rmg_coincident_nodes_1(self, grid, nmgrid_c):
        """coarse-scale network model grid, remove duplicates"""
        link_nodes_c = gt.get_link_nodes(nmgrid_c)
        nmg_link_to_rmg_coincident_nodes_mapper = (
            gt.map_nmg_links_to_rmg_coincident_nodes(
                grid, nmgrid_c, link_nodes_c, remove_duplicates=True
            )
        )

        nmg_link_to_rmg_coincident_nodes_mapper_e = np.array(
            [
                [0.0, 3.0, 30.0, 0.0, 31.6227766, 2300.0],
                [0.0, 10.0, 30.0, 10.0, 26.33648662, 2300.0],
                [0.0, 16.0, 20.0, 20.0, 15.79556109, 2300.0],
                [0.0, 23.0, 20.0, 30.0, 5.25463555, 2300.0],
                [1.0, 30.0, 20.0, 40.0, 4.99499499, 1100.0],
                [2.0, 24.0, 30.0, 30.0, 16.76491407, 1050.0],
                [2.0, 31.0, 30.0, 40.0, 11.16914836, 1050.0],
                [2.0, 32.0, 40.0, 40.0, 5.57338265, 1050.0],
            ]
        )

        check_vals(
            np.array(list(nmg_link_to_rmg_coincident_nodes_mapper.values())).T,
            nmg_link_to_rmg_coincident_nodes_mapper_e,
        )

        # check no duplicate nodes
        assert len(
            np.unique(nmg_link_to_rmg_coincident_nodes_mapper["coincident_node"])
        ) == len(nmg_link_to_rmg_coincident_nodes_mapper["coincident_node"])

    def test_map_nmg_links_to_rmg_coincident_nodes_2(self, grid, nmgrid_c):
        """coarse-scale network model grid, don't remove duplicate nodes"""
        link_nodes_c = gt.get_link_nodes(nmgrid_c)
        nmg_link_to_rmg_coincident_nodes_mapper_dup = (
            gt.map_nmg_links_to_rmg_coincident_nodes(
                grid, nmgrid_c, link_nodes_c, remove_duplicates=False
            )
        )

        nmg_link_to_rmg_coincident_nodes_mapper_dup_e = np.array(
            [
                [0.0, 3.0, 30.0, 0.0, 31.6227766, 2300.0],
                [0.0, 10.0, 30.0, 10.0, 26.33648662, 2300.0],
                [0.0, 16.0, 20.0, 20.0, 15.79556109, 2300.0],
                [0.0, 23.0, 20.0, 30.0, 5.25463555, 2300.0],
                [1.0, 23.0, 20.0, 30.0, 10.0, 1100.0],
                [1.0, 30.0, 20.0, 40.0, 4.99499499, 1100.0],
                [2.0, 23.0, 20.0, 30.0, 22.36067977, 1050.0],
                [2.0, 24.0, 30.0, 30.0, 16.76491407, 1050.0],
                [2.0, 31.0, 30.0, 40.0, 11.16914836, 1050.0],
                [2.0, 32.0, 40.0, 40.0, 5.57338265, 1050.0],
            ]
        )

        check_vals(
            np.array(list(nmg_link_to_rmg_coincident_nodes_mapper_dup.values())).T,
            nmg_link_to_rmg_coincident_nodes_mapper_dup_e,
        )

        # check that duplicate nodes are included
        assert len(
            np.unique(nmg_link_to_rmg_coincident_nodes_mapper_dup["coincident_node"])
        ) != len(nmg_link_to_rmg_coincident_nodes_mapper_dup["coincident_node"])

    def test_map_nmg_links_to_rmg_coincident_nodes_3(self, grid, nmgrid_f):
        """fine-scale network model grid, remove duplicates"""
        link_nodes_f = gt.get_link_nodes(nmgrid_f)
        nmg_link_to_rmg_coincident_nodes_mapper = (
            gt.map_nmg_links_to_rmg_coincident_nodes(
                grid, nmgrid_f, link_nodes_f, remove_duplicates=True
            )
        )

        nmg_link_to_rmg_coincident_nodes_mapper_e = np.array(
            [
                [0, 3, 30.0, 0.0, 14.142136, 2750.0],
                [0, 9, 20.0, 10.0, 7.063990, 2750.0],
                [1, 17, 30.0, 20.0, 7.063990, 2400.0],
                [2, 23, 20.0, 30.0, 7.063990, 1950.0],
                [3, 30, 20.0, 40.0, 4.994995, 1100.0],
            ]
        )

        check_vals(
            np.array(list(nmg_link_to_rmg_coincident_nodes_mapper.values())).T[:5],
            nmg_link_to_rmg_coincident_nodes_mapper_e,
        )

        # check no duplicate nodes
        assert len(
            np.unique(nmg_link_to_rmg_coincident_nodes_mapper["coincident_node"])
        ) == len(nmg_link_to_rmg_coincident_nodes_mapper["coincident_node"])

    def test_map_nmg_links_to_rmg_coincident_nodes_4(self, grid_up, nmgrid_c_up):
        """upside down coarse-scale network model grid, remove duplicates"""
        link_nodes_c_up = gt.get_link_nodes(nmgrid_c_up)
        nmg_link_to_rmg_coincident_nodes_mapper_up = (
            gt.map_nmg_links_to_rmg_coincident_nodes(
                grid_up, nmgrid_c_up, link_nodes_c_up, remove_duplicates=True
            )
        )

        nmg_link_to_rmg_coincident_nodes_mapper_up_e = np.array(
            [
                [0.0, 31.0, 30.0, 40.0, 16.76491407, 1050.0],
                [0.0, 24.0, 30.0, 30.0, 11.16914836, 1050.0],
                [0.0, 23.0, 20.0, 30.0, 5.57338265, 1050.0],
                [1.0, 25.0, 40.0, 30.0, 4.99499499, 1100.0],
                [2.0, 52.0, 30.0, 70.0, 31.6227766, 2300.0],
                [2.0, 45.0, 30.0, 60.0, 26.33648662, 2300.0],
                [2.0, 39.0, 40.0, 50.0, 15.79556109, 2300.0],
                [2.0, 32.0, 40.0, 40.0, 5.25463555, 2300.0],
            ]
        )

        check_vals(
            np.array(list(nmg_link_to_rmg_coincident_nodes_mapper_up.values())).T,
            nmg_link_to_rmg_coincident_nodes_mapper_up_e,
        )

        # check no duplicate nodes
        assert len(
            np.unique(nmg_link_to_rmg_coincident_nodes_mapper_up["coincident_node"])
        ) == len(nmg_link_to_rmg_coincident_nodes_mapper_up["coincident_node"])

    def test_map_nmg_links_to_rmg_coincident_nodes_5(self, grid, nmgrid_s):
        """single-channel-reach network model grid, remove duplicates"""
        link_nodes_s = gt.get_link_nodes(nmgrid_s)
        nmg_link_to_rmg_coincident_nodes_mapper_s = (
            gt.map_nmg_links_to_rmg_coincident_nodes(
                grid, nmgrid_s, link_nodes_s, remove_duplicates=True
            )
        )

        nmg_link_to_rmg_coincident_nodes_mapper_s_e = np.array(
            [
                [0.0, 3.0, 30.0, 0.0, 31.6227766, 2300.0],
                [0.0, 10.0, 30.0, 10.0, 26.33648662, 2300.0],
                [0.0, 16.0, 20.0, 20.0, 15.79556109, 2300.0],
                [0.0, 23.0, 20.0, 30.0, 5.25463555, 2300.0],
            ]
        )

        check_vals(
            np.array(list(nmg_link_to_rmg_coincident_nodes_mapper_s.values())).T,
            nmg_link_to_rmg_coincident_nodes_mapper_s_e,
        )

        # check no duplicate nodes
        assert len(
            np.unique(nmg_link_to_rmg_coincident_nodes_mapper_s["coincident_node"])
        ) == len(nmg_link_to_rmg_coincident_nodes_mapper_s["coincident_node"])

    def test_map_nmg_links_to_rmg_coincident_nodes_6(self, grid_up, nmgrid_s_up):
        """upside down single-channel-reach network model grid, remove duplicates"""
        link_nodes_s_up = gt.get_link_nodes(nmgrid_s_up)
        nmg_link_to_rmg_coincident_nodes_mapper_s_up = (
            gt.map_nmg_links_to_rmg_coincident_nodes(
                grid_up, nmgrid_s_up, link_nodes_s_up, remove_duplicates=True
            )
        )

        nmg_link_to_rmg_coincident_nodes_mapper_s_up_e = np.array(
            [
                [0.0, 52.0, 30.0, 70.0, 31.6227766, 2300.0],
                [0.0, 45.0, 30.0, 60.0, 26.33648662, 2300.0],
                [0.0, 39.0, 40.0, 50.0, 15.79556109, 2300.0],
                [0.0, 32.0, 40.0, 40.0, 5.25463555, 2300.0],
            ]
        )

        check_vals(
            np.array(list(nmg_link_to_rmg_coincident_nodes_mapper_s_up.values())).T,
            nmg_link_to_rmg_coincident_nodes_mapper_s_up_e,
        )

        # check no duplicate nodes
        assert len(
            np.unique(nmg_link_to_rmg_coincident_nodes_mapper_s_up["coincident_node"])
        ) == len(nmg_link_to_rmg_coincident_nodes_mapper_s_up["coincident_node"])
        
        
class TestMapRMGNodesToNMGLinks: 
    
    def test_map_rmg_nodes_to_nmg_links_1(self, grid):
        """normal test"""
        # provide a link to rmg coicident node mapper
        nmg_link_to_rmg_coincident_nodes_mapper = {'linkID': np.array([0, 0, 0, 0, 1, 2, 2, 2]),
                                                    'coincident_node': np.array([ 3, 10, 16, 23, 30, 24, 31, 32]),
                                                    'x': np.array([30., 30., 20., 20., 20., 30., 30., 40.]),
                                                    'y': np.array([ 0., 10., 20., 30., 40., 30., 40., 40.]),
                                                    'coincident_node_downstream_dist': np.array([31.6227766 , 26.33648662, 15.79556109,  5.25463555,  4.99499499,
                                                            16.76491407, 11.16914836,  5.57338265]),
                                                    'link_drainage_area': np.array([2300., 2300., 2300., 2300., 1100., 1050., 1050., 1050.])}
        # channel node array that will be mapped to the nmg links
        acn = np.array([ 3,  9, 17, 23, 30, 31, 32, 36])
        # get the node to nmg link mapper
        cn_to_nmg_link_mapper = gt.map_rmg_nodes_to_nmg_links(grid, nmg_link_to_rmg_coincident_nodes_mapper, acn)
        
        l0_e = [np.int64(3), np.int64(9), np.int64(17), np.int64(23)]
        l1_e = [np.int64(30), np.int64(36)]
        l2_e = [np.int64(31), np.int64(32)]

        check_vals(cn_to_nmg_link_mapper['node'][cn_to_nmg_link_mapper['linkID'] == 0], l0_e)
        check_vals(cn_to_nmg_link_mapper['node'][cn_to_nmg_link_mapper['linkID'] == 1], l1_e)
        check_vals(cn_to_nmg_link_mapper['node'][cn_to_nmg_link_mapper['linkID'] == 2], l2_e)


        # check that each channel and terrace node has only one equivalent nmg rmg node.
        assert len(np.unique(cn_to_nmg_link_mapper['node'])) == len(cn_to_nmg_link_mapper['node'])
     
    def test_map_rmg_nodes_to_nmg_links_2(self, grid_up):
        """same test, using upside down grid"""
        # provide a link to rmg coicident node mapper
        nmg_link_to_rmg_coincident_nodes_mapper_up = {'linkID': np.array([0, 0, 0, 1, 2, 2, 2, 2]),
                                                    'coincident_node': np.array([31, 24, 23, 25, 52, 45, 39, 32]),
                                                    'x': np.array([30., 30., 20., 40., 30., 30., 40., 40.]),
                                                    'y': np.array([40., 30., 30., 30., 70., 60., 50., 40.]),
                                                    'coincident_node_downstream_dist': np.array([16.76491407, 11.16914836,  5.57338265,  4.99499499, 31.6227766 ,
                                                            26.33648662, 15.79556109,  5.25463555]),
                                                    'link_drainage_area': np.array([1050., 1050., 1050., 1100., 2300., 2300., 2300., 2300.])}
        # channel node array that will be mapped to the nmg links
        acn_up = np.array([19, 23, 24, 25, 32, 38, 46, 52])
        
        cn_to_nmg_link_mapper_up = gt.map_rmg_nodes_to_nmg_links(grid_up, nmg_link_to_rmg_coincident_nodes_mapper_up, acn_up)

        l0_e = [np.int64(23), np.int64(24)]
        l1_e = [np.int64(19), np.int64(25)]
        l2_e = [np.int64(32), np.int64(38),np.int64(46), np.int64(52)]

        check_vals(cn_to_nmg_link_mapper_up['node'][cn_to_nmg_link_mapper_up['linkID'] == 0], l0_e)
        check_vals(cn_to_nmg_link_mapper_up['node'][cn_to_nmg_link_mapper_up['linkID'] == 1], l1_e)
        check_vals(cn_to_nmg_link_mapper_up['node'][cn_to_nmg_link_mapper_up['linkID'] == 2], l2_e)


        # check that each channel and terrace node has only one equivalent nmg rmg node.
        assert len(np.unique(cn_to_nmg_link_mapper_up['node'])) == len(cn_to_nmg_link_mapper_up['node'])
        
    def test_map_rmg_nodes_to_nmg_links_3(self, grid_up):        
        """for upside down grid, but with remove_small_tribs = 10 (shouldn't affect results)"""
        # provide a link to rmg coicident node mapper
        nmg_link_to_rmg_coincident_nodes_mapper_up = {'linkID': np.array([0, 0, 0, 1, 2, 2, 2, 2]),
                                                        'coincident_node': np.array([31, 24, 23, 25, 52, 45, 39, 32]),
                                                        'x': np.array([30., 30., 20., 40., 30., 30., 40., 40.]),
                                                        'y': np.array([40., 30., 30., 30., 70., 60., 50., 40.]),
                                                        'coincident_node_downstream_dist': np.array([16.76491407, 11.16914836,  5.57338265,  4.99499499, 31.6227766 ,
                                                                26.33648662, 15.79556109,  5.25463555]),
                                                        'link_drainage_area': np.array([1050., 1050., 1050., 1100., 2300., 2300., 2300., 2300.])}
        # channel node array that will be mapped to the nmg links
        acn_up = np.array([19, 23, 24, 25, 32, 38, 46, 52])
        
        cn_to_nmg_link_mapper_up = gt.map_rmg_nodes_to_nmg_links(grid_up, nmg_link_to_rmg_coincident_nodes_mapper_up, acn_up, remove_small_trib_ratio = 1/10)

        l0_e = [np.int64(23), np.int64(24)]
        l1_e = [np.int64(19), np.int64(25)]
        l2_e = [np.int64(32), np.int64(38),np.int64(46), np.int64(52)]

        check_vals(cn_to_nmg_link_mapper_up['node'][cn_to_nmg_link_mapper_up['linkID'] == 0], l0_e)
        check_vals(cn_to_nmg_link_mapper_up['node'][cn_to_nmg_link_mapper_up['linkID'] == 1], l1_e)
        check_vals(cn_to_nmg_link_mapper_up['node'][cn_to_nmg_link_mapper_up['linkID'] == 2], l2_e)

        # check that each channel and terrace node has only one equivalent nmg rmg node.
        assert len(np.unique(cn_to_nmg_link_mapper_up['node'])) == len(cn_to_nmg_link_mapper_up['node'])

    
class TestRemoveSmallTribs:
    def test_remove_small_tribs_1(self):
        nmg_link_to_rmg_coincident_nodes_mapper = {'linkID': np.array([0, 0, 0, 0, 1, 2, 2, 2]),
                                                    'coincident_node': np.array([ 3, 10, 16, 23, 30, 24, 31, 32]),
                                                    'x': np.array([30., 30., 20., 20., 20., 30., 30., 40.]),
                                                    'y': np.array([ 0., 10., 20., 30., 40., 30., 40., 40.]),
                                                    'coincident_node_downstream_dist': np.array([31.6227766 , 26.33648662, 15.79556109,  5.25463555,  4.99499499,
                                                            16.76491407, 11.16914836,  5.57338265]),
                                                    'link_drainage_area': np.array([2300., 2300., 2300., 2300., 1100., 1050., 1050., 1050.])}
        rmg_nodes_to_nmg_links_mapper = {'node': np.array([ 3,  9, 17, 23]),
                                        'linkID': np.array([0, 0, 0, 0]),
                                        'coincident_node_downstream_dist': np.array([31.6227766 , 26.33648662, 26.33648662,  5.25463555]),
                                        'coincident_node': np.array([ 3, 10, 16, 23]),
                                        'link_drainage_area': np.array([2300., 2300., 2300., 2300.]),
                                        'node_drainage_area': np.array([2900., 2300.,  100., 1600.])}
        # remove the small trib from rmg_nodes_to_nmg_links_mapper
        rmg_nodes_to_nmg_links_mapper_corrected = gt._remove_small_tribs(rmg_nodes_to_nmg_links_mapper, nmg_link_to_rmg_coincident_nodes_mapper, remove_small_trib_ratio = 1/10)# cn_to_nmg_link_mapper)


        rmg_nodes_to_nmg_links_mapper_e = np.array([[   3.        ,    9.        ,   23.        ],
               [   0.        ,    0.        ,    0.        ],
               [  31.6227766 ,   26.33648662,    5.25463555],
               [   3.        ,   10.        ,   23.        ],
               [2300.        , 2300.        , 2300.        ],
               [2900.        , 2300.        , 1600.        ]])
        
        check_vals(np.array(list(rmg_nodes_to_nmg_links_mapper_corrected.values())),
                       rmg_nodes_to_nmg_links_mapper_e)
        
    def test_remove_small_tribs_2(self):
        
        
    def test_remove_small_tribs_3(self):
    
        
    def test_remove_small_tribs_4(self):
        
        