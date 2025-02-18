import matplotlib.pyplot as plt
import numpy as np
import pydantic.v1 as pydantic
import pytest
import responses
import tidy3d as td
from tidy3d.exceptions import SetupError, ValidationError
from tidy3d.plugins.dispersion import (
    AdvancedFastFitterParam,
    DispersionFitter,
    FastDispersionFitter,
)
from tidy3d.plugins.dispersion.web import run as run_fitter

advanced_param = AdvancedFastFitterParam(num_iters=1, passivity_num_iters=1)


_, AX = plt.subplots()


@pytest.fixture
def random_data():
    data_points = 11
    wvl_um = np.linspace(1, 2, data_points)
    n_data = np.random.random(data_points)
    k_data = np.random.random(data_points)
    return wvl_um, n_data, k_data


@pytest.fixture
def mock_remote_api(monkeypatch):
    def mock_url(*args, **kwargs):
        return "http://monkeypatched.com"

    monkeypatch.setattr("tidy3d.plugins.dispersion.web.FitterData._set_url", mock_url)
    responses.add(responses.GET, f"{mock_url()}/health", status=200)
    responses.add(
        responses.POST,
        f"{mock_url()}/dispersion/fit",
        json={"message": td.PoleResidue().json(), "rms": 1e-16},
        status=200,
    )


def test_coeffs():
    """make sure pack_coeffs and unpack_coeffs are reciprocal"""
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    a, c = DispersionFitter._unpack_coeffs(coeffs)
    coeffs_ = DispersionFitter._pack_coeffs(a, c)
    a_, c_ = DispersionFitter._unpack_coeffs(coeffs_)
    assert np.allclose(coeffs, coeffs_)
    assert np.allclose(a, a_)
    assert np.allclose(c, c_)

    assert np.isclose(DispersionFitter._eV_to_Hz(DispersionFitter._Hz_to_eV(1.0)), 1.0)


def test_pole_coeffs():
    """make sure coeffs_to_poles and poles_to_coeffs are reciprocal"""
    num_poles = 10
    coeffs = np.random.random(4 * num_poles)
    poles = DispersionFitter._coeffs_to_poles(coeffs)
    coeffs_ = DispersionFitter._poles_to_coeffs(poles)
    poles_ = DispersionFitter._coeffs_to_poles(coeffs_)
    assert np.allclose(coeffs, coeffs_)
    assert np.allclose(poles, poles_)


@responses.activate
def test_lossless_dispersion(random_data, mock_remote_api):
    """perform fitting on random data"""

    # wrong input data
    with pytest.raises(pydantic.ValidationError):
        fitter = DispersionFitter(wvl_um=[], n_data=())

    with pytest.raises(pydantic.ValidationError):
        fitter = DispersionFitter(wvl_um=[1.0], n_data=(1.0, 1.1))

    with pytest.raises(pydantic.ValidationError):
        fitter = DispersionFitter(wvl_um=[1.0], n_data=(1.0), k_data=(0, 1))

    with pytest.raises(SetupError):
        fitter = DispersionFitter(wvl_um=[1.0], n_data=(1.0), wvl_range=(2, 3))
        medium, rms = fitter.fit(num_tries=2)

    wvl_um, n_data, _ = random_data
    fitter = DispersionFitter(wvl_um=wvl_um.tolist(), n_data=tuple(n_data))
    medium, rms = fitter._fit_single()
    medium, rms = fitter.fit(num_tries=2)
    medium, rms = run_fitter(fitter)

    fitter = FastDispersionFitter(wvl_um=wvl_um.tolist(), n_data=tuple(n_data))
    medium, rms = fitter.fit(advanced_param=advanced_param)

    # from permittivity data
    fitter = FastDispersionFitter.from_complex_permittivity(wvl_um, n_data**2)
    medium2, rms2 = fitter.fit(advanced_param=advanced_param)


@responses.activate
def test_lossy_dispersion(random_data, mock_remote_api):
    """perform fitting on random lossy data"""
    wvl_um, n_data, k_data = random_data
    fitter = DispersionFitter(wvl_um=wvl_um, n_data=n_data, k_data=k_data)
    medium, rms = fitter._fit_single()
    medium, rms = fitter.fit(num_tries=2)
    medium, rms = run_fitter(fitter)

    fitter = FastDispersionFitter(wvl_um=wvl_um.tolist(), n_data=n_data, k_data=k_data)
    medium, rms = fitter.fit(advanced_param=advanced_param)

    # from permittivity data
    eps_complex = (n_data + 1j * k_data) ** 2
    fitter = FastDispersionFitter.from_complex_permittivity(
        wvl_um, eps_complex.real, eps_complex.imag
    )
    medium2, rms2 = fitter.fit(advanced_param=advanced_param)

    # from loss tangent
    fitter = FastDispersionFitter.from_loss_tangent(
        wvl_um, eps_complex.real, eps_complex.imag / eps_complex.real
    )
    medium3, rms3 = fitter.fit(advanced_param=advanced_param)

    # test that poles can be close but not exactly equal to provided freqs
    N = 2
    wvl_um = np.linspace(0.5, 0.6, N)
    n_data = np.ones(N) * 2
    k_data = np.ones(N) * 0.5

    fitter = FastDispersionFitter(wvl_um=wvl_um, n_data=n_data, k_data=k_data)
    medium, rms_error = fitter.fit(max_num_poles=2, tolerance_rms=1e-3)


def test_constant_loss_tangent():
    """perform fitting on constant loss tangent material"""

    eps_real = 2.5
    loss_tangent = 1e-2
    frequency_range = (1e9, 6e9)
    mat = FastDispersionFitter.constant_loss_tangent_model(eps_real, loss_tangent, frequency_range)

    # validate
    sampling_frequency = np.linspace(frequency_range[0], frequency_range[1], 29)
    eps_out, loss_tangent_out = mat.loss_tangent_model(sampling_frequency)
    assert np.max(np.abs(eps_out - eps_real)) < 2e-2
    assert np.max(np.abs(loss_tangent_out - loss_tangent)) / loss_tangent < 2e-2


@responses.activate
def test_dispersion_load_url():
    """loads dispersion model from url"""

    def _test_nk(mock_data):
        responses.add(responses.GET, "http://test.com/nk_data.csv", body=b"\n".join(mock_data))
        return DispersionFitter.from_url("http://test.com/nk_data.csv")

    # doesn't start with "wl, n"
    with pytest.raises(ValidationError):
        mock_data = [b"wavelength,n", b"1,2", b"3,1*"]
        _test_nk(mock_data)

    # contains strings other than "wl,n", "wl,k"
    with pytest.raises(ValidationError):
        mock_data = [b"wl,n", b"1,2", b"3,1", b"wl,loss", b"1,2", b"3,1*"]
        _test_nk(mock_data)

    # mixed symbols with numbers
    with pytest.raises(ValidationError):
        mock_data = [b"wl,n", b"1,2", b"3,1*"]
        _test_nk(mock_data)

    # number of n/k data unmatched
    with pytest.raises(ValidationError):
        mock_data = [b"wl,n", b"1,2", b"3,2.1", b"wl,k", b"1,0"]
        _test_nk(mock_data)

    # has k data more than once
    with pytest.raises(ValidationError):
        mock_data = [b"wl,n", b"1,2", b"3,2.1", b"wl,k", b"1,0", b"wl,k", b"1,0"]
        _test_nk(mock_data)

    # n only
    mock_data = [b"wl,n", b"1,2", b"2,2"]
    fitter = _test_nk(mock_data)
    medium, rms = fitter.fit(num_tries=10)

    # n and k
    mock_data = [b"wl,n", b"1,2", b"3,2.1", b"wl,k", b"1,0", b"3,0.1"]
    fitter = _test_nk(mock_data)
    medium, rms = fitter.fit(num_tries=2)


def test_dispersion_load_file():
    """loads dispersion model from nk data file"""

    fitter = DispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=2)

    fitter = DispersionFitter.from_file("tests/data/n_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(num_tries=20)

    fitter = FastDispersionFitter.from_file("tests/data/nk_data.csv", skiprows=1, delimiter=",")
    medium, rms = fitter.fit(advanced_param=advanced_param)


def test_dispersion_plot(random_data):
    """plots a medium fit from file"""
    wvl_um, n_data, k_data = random_data

    fitter = DispersionFitter(wvl_um=wvl_um, n_data=n_data)
    fitter.plot(ax=AX)
    plt.close()
    medium, rms = fitter.fit(num_tries=2)
    fitter.plot(medium, ax=AX)
    plt.close()

    fitter = DispersionFitter(wvl_um=wvl_um, n_data=n_data, k_data=k_data)
    fitter.plot()
    plt.close()
    medium, rms = fitter.fit(num_tries=2)
    fitter.plot(medium, ax=AX)
    plt.close()


def test_dispersion_set_wvg_range(random_data):
    """set wavelength range function"""
    wvl_um, n_data, k_data = random_data
    fitter = DispersionFitter(wvl_um=wvl_um, n_data=n_data)
    fastfitter = FastDispersionFitter(wvl_um=wvl_um, n_data=n_data)

    wvl_range = [1.2, 1.8]
    fitter = fitter.copy(update={"wvl_range": wvl_range})
    assert len(fitter.freqs) == 7
    medium, rms = fitter.fit(num_tries=2)
    fastfitter = fastfitter.copy(update={"wvl_range": wvl_range})
    assert len(fastfitter.freqs) == 7
    medium, rms = fastfitter.fit(advanced_param=advanced_param)

    wvl_range = [1.2, 2.8]
    fitter = fitter.copy(update={"wvl_range": wvl_range, "k_data": k_data})
    assert len(fitter.freqs) == 9
    medium, rms = fitter.fit(num_tries=2)
    fastfitter = fastfitter.copy(update={"wvl_range": wvl_range})
    assert len(fastfitter.freqs) == 9
    medium, rms = fastfitter.fit(advanced_param=advanced_param)

    wvl_range = [0.2, 1.8]
    fitter = fitter.copy(update={"wvl_range": wvl_range})
    assert len(fitter.freqs) == 9
    medium, rms = fitter.fit(num_tries=2)
    fastfitter = fastfitter.copy(update={"wvl_range": wvl_range})
    assert len(fastfitter.freqs) == 9
    medium, rms = fastfitter.fit(advanced_param=advanced_param)

    wvl_range = [0.2, 2.8]
    fitter = fitter.copy(update={"wvl_range": wvl_range, "k_data": k_data})
    assert len(fitter.freqs) == 11
    medium, rms = fitter.fit(num_tries=2)
    fastfitter = fastfitter.copy(update={"wvl_range": wvl_range})
    assert len(fastfitter.freqs) == 11
    medium, rms = fastfitter.fit(advanced_param=advanced_param)


def test_dispersion_guess(random_data):
    """plots a medium fit from file"""
    wvl_um, n_data, k_data = random_data

    fitter = DispersionFitter(wvl_um=wvl_um, n_data=n_data)
    medium, rms = fitter.fit(num_tries=2)

    medium_new, rms_new = fitter.fit(num_tries=1, guess=medium)


def test_dispersion_loss_samples():
    wvls = np.array([275e-3, 260e-3, 255e-3])
    n_nAlGaN = np.array([2.72, 2.68, 2.53])

    nAlGaN_fitter = FastDispersionFitter(wvl_um=wvls, n_data=n_nAlGaN)
    nAlGaN_mat, _ = nAlGaN_fitter.fit()

    freq_list = nAlGaN_mat.angular_freq_to_Hz(nAlGaN_mat._imag_ep_extrema_with_samples())
    ep = nAlGaN_mat.eps_model(freq_list)
    for e in ep:
        assert e.imag >= 0
