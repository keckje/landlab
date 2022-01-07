import pytest
import numpy as np

from landlab.utils.parcels import SedimentPulserEachParcel, SedimentPulserAtLinks

from landlab import RasterModelGrid

from landlab.utils.parcels import (
                                    calc_total_parcel_volume,
                                    calc_lognormal_distribution_parameters,
                                    )



def test_call_default_inputs(example_nmg, example_parcels):
    """call the SedimentPulserAtLinnks with default inputs"""
    PulserAtLinks = SedimentPulserAtLinks(example_nmg, parcels = example_parcels)
    _ = PulserAtLinks(time = 1)
    
    
def test_call_with_time(example_nmg, example_parcels):
    """call the SedimentPulserAtLinnks with time_pulse_defined, but time
    is not equal to time_to_pulse"""
    time = 2
    PulserAtLinks = SedimentPulserAtLinks(example_nmg, parcels = example_parcels,
                                          time_to_pulse = lambda time: time == 2)
    _ = PulserAtLinks(time = 1)


def test_call_with_time_at_pulse_time(example_nmg, example_parcels):
    """call the SedimentPulserAtLinnks with time_pulse_defined, but time
    is not equal to time_to_pulse"""
    time = 2
    PulserAtLinks = SedimentPulserAtLinks(example_nmg, parcels = example_parcels,
                                          time_to_pulse = lambda time: time == 2)
    _ = PulserAtLinks(time = 2)
    
    
