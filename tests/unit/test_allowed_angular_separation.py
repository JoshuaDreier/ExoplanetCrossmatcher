import astropy.units as u
import numpy as np
import pytest

from crossmatching import allowed_angular_separation


def test_known_pm_and_epoch_uses_computed_radius():
    # radius = (pm + pm_err) * |epoch - input_epoch| + minimum = (1.0 + 0.5)*10 + 10
    radii = allowed_angular_separation(
        proper_motion=np.array([1.0]),
        pm_err=np.array([0.5]),
        epoch=np.array([2010.0]),
        input_epoch=2000
    )
    assert radii[0].to_value(u.arcsec) == pytest.approx(25.0)


def test_masked_pm_or_epoch_gets_unknown_default():
    pm    = np.ma.MaskedArray([1.0, 1.0], mask=[True, False])
    pmerr = np.ma.MaskedArray([0.5, 0.5], mask=[False, False])
    epoch = np.ma.MaskedArray([2010.0, 2010.0], mask=[False, True])
    radii = allowed_angular_separation(pm, pmerr, epoch, input_epoch=2000, unknown_default=42 * u.arcsec)
    assert radii[0].to_value(u.arcsec) == pytest.approx(42.0)
    assert radii[1].to_value(u.arcsec) == pytest.approx(42.0)


def test_masked_pm_err_contributes_zero_not_unknown_default():
    # Known pm but unknown pm_err: the row must keep its computed radius
    # (pm_err treated as 0), not fall back to unknown_default.
    pm    = np.ma.MaskedArray([1.0], mask=[False])
    pmerr = np.ma.MaskedArray([0.5], mask=[True])
    epoch = np.ma.MaskedArray([2010.0], mask=[False])
    radii = allowed_angular_separation(pm, pmerr, epoch, input_epoch=2000, unknown_default=50 * u.arcsec)
    assert radii[0].to_value(u.arcsec) == pytest.approx(1.0 * 10 + 10)
