"""
Tests in here check the emittance evolution calculation for both NagaitsevIBS and BjorkenMtingwaIBS.
The simple and fast case of the PS protons at injection is taken to compute the growth rates used.
"""
import numpy as np
import xtrack as xt
import xpart as xp
from cpymad.madx import Madx

from xibs._old_michalis import MichalisIBS
from xibs.analytical import BjorkenMtingwaIBS, IBSGrowthRates, NagaitsevIBS, _SynchrotronRadiationInputs
from xibs.inputs import BeamParameters, OpticsParameters


def test_emittance_evolution(madx_ps_injection_protons, xtrack_ps_injection_protons):
    # --------------------------------------------------------------------
    # Get the growth rates from MAD-X
    madx, params = madx_ps_injection_protons  # fully set up from the config file
    # --------------------------------------------------------------------
    # Load the optics, from MAD-X for maximum consistency
    twiss = madx.twiss(centre=True).dframe()
    seq_name = madx.table.twiss.summary.sequence  # works whichever your sequence
    frev_hz = madx.sequence[seq_name].beam.freq0 * 1e6  # beware freq0 is in MHz
    gamma_rel = madx.sequence[seq_name].beam.gamma  # relativistic gamma
    gamma_tr = madx.table.summ.gammatr[0]  # transition gamma
    slipfactor = (1 / (gamma_tr**2)) - (1 / (gamma_rel**2))  # use the xsuite convention!
    opticsparams = OpticsParameters(twiss, slipfactor, frev_hz)
    # --------------------------------------------------------------------
    # Load the beam parameters from the equivalent xsuite line
    line: xt.Line = xtrack_ps_injection_protons
    beamparams = BeamParameters(line.particle_ref)
    beamparams.n_part = madx.sequence[seq_name].beam.npart  # as line.particle_ref is only 1 particle
    # --------------------------------------------------------------------
    # Set hardcoded growth rates - faster test this way - we don't care about being
    # realistic here, just that with these specific values the computation is correct
    rates = IBSGrowthRates(Tx=-8.534561002903093e-05, Ty=2.444062101488359e-05, Tz=0.00033520513993030626)
    NIBS = NagaitsevIBS(beamparams, opticsparams)
    NIBS.ibs_growth_rates = rates
    BMIBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    BMIBS.ibs_growth_rates = rates
    # --------------------------------------------------------------------
    # Compute new emittances without specifying the time step - it will fallback to revolution frequency
    # We compare to expected values, for NagaitsevIBS and BjorkenMtingwaIBS
    epsx, epsy, sigd, bl = NIBS.emittance_evolution(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length
    )
    assert np.isclose(epsx, 2.685585551672946e-07, atol=0)
    assert np.isclose(epsy, 1.6784909702029845e-07, atol=0)
    assert np.isclose(sigd, 0.0002740293585944697, atol=0)
    assert np.isclose(bl, 0.5684116449445993, atol=0)
    epsx, epsy, sigd, bl = BMIBS.emittance_evolution(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length
    )
    assert np.isclose(epsx, 2.685585551672946e-07, atol=0)
    assert np.isclose(epsy, 1.6784909702029845e-07, atol=0)
    assert np.isclose(sigd, 0.0002740293585944697, atol=0)
    assert np.isclose(bl, 0.5684116449445993, atol=0)
    # --------------------------------------------------------------------
    # Compute new emittances this time when specifying the time step (arbitrary number is fine)
    # We compare to expected values, for NagaitsevIBS and BjorkenMtingwaIBS
    epsx, epsy, sigd, bl = NIBS.emittance_evolution(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length, dt=1
    )
    assert np.isclose(epsx, 2.685356359022883e-07, atol=0)
    assert np.isclose(epsy, 1.6785319939752902e-07, atol=0)
    assert np.isclose(sigd, 0.0002740752903667045, atol=0)
    assert np.isclose(bl, 0.5685069199704035, atol=0)
    epsx, epsy, sigd, bl = BMIBS.emittance_evolution(
        params.geom_epsx, params.geom_epsy, params.sig_delta, params.bunch_length, dt=1
    )
    assert np.isclose(epsx, 2.685356359022883e-07, atol=0)
    assert np.isclose(epsy, 1.6785319939752902e-07, atol=0)
    assert np.isclose(sigd, 0.0002740752903667045, atol=0)
    assert np.isclose(bl, 0.5685069199704035, atol=0)


def test_emittance_evolution_with_synchrotron_radiation(xtrack_clic_damping_ring):
    """Test emittance evolution with SR. Compare to Michalis results."""
    # --------------------------------------------------------------------
    # A whole setup to get initial values, same as in the CLIC DR benchmark
    bunch_intensity = 4.4e9
    sigma_z = 1.58e-3
    nemitt_x = 5.6644e-07
    nemitt_y = 3.7033e-09
    n_part = int(5e3)
    line = xtrack_clic_damping_ring
    line.particle_ref = xp.Particles(mass0=xp.ELECTRON_MASS_EV, q0=1, p0c=2.86e9)
    for cavity in [element for element in line.elements if isinstance(element, xt.Cavity)]:
        cavity.lag = 180
    twiss = line.twiss()
    particles = xp.generate_matched_gaussian_bunch(
        num_particles=n_part,
        total_intensity_particles=bunch_intensity,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        line=line,
    )
    sig_x = np.std(particles.x[particles.state > 0])  # horizontal stdev
    sig_y = np.std(particles.y[particles.state > 0])  # vertical stdev
    sigma_delta = np.std(particles.delta[particles.state > 0])  # momentum spread
    geom_epsx = (sig_x**2 - (twiss["dx"][0] * sigma_delta) ** 2) / twiss["betx"][0]
    geom_epsy = (sig_y**2 - (twiss["dy"][0] * sigma_delta) ** 2) / twiss["bety"][0]
    bunch_length = np.std(particles.zeta[particles.state > 0])
    # --------------------------------------------------------------------
    # Create input parameters and IBS classes
    opticsparams = OpticsParameters(twiss)
    beamparams = BeamParameters(particles)
    NIBS = NagaitsevIBS(beamparams, opticsparams)
    BMIBS = BjorkenMtingwaIBS(beamparams, opticsparams)
    MIBS = MichalisIBS()
    MIBS.set_beam_parameters(particles)
    MIBS.set_optic_functions(twiss)
    # --------------------------------------------------------------------
    # Set hardcoded growth rates - faster test this way - we don't care about being
    # realistic here, just that with these specific values the computation is correct
    rates = IBSGrowthRates(Tx=1e-6, Ty=1e-6, Tz=1e-5)
    NIBS.ibs_growth_rates = rates
    BMIBS.ibs_growth_rates = rates
    MIBS.Ixx = rates.Tx; MIBS.Iyy = rates.Ty; MIBS.Ipp = rates.Tz
    # --------------------------------------------------------------------
    # Get the SR inputs needed for the emittance evolution
    sr_inputs = _get_sr_inputs_from_line(line)
    # --------------------------------------------------------------------
    # Compute ref emittances at next turn, aka using revolution time as time step
    # We compare to expected values, for NagaitsevIBS and BjorkenMtingwaIBS
    time_step = 1 / opticsparams.revolution_frequency
    ref_epsx, ref_epsy, ref_sigma_delta = MIBS.emit_evol_with_SR(
        geom_epsx, 
        geom_epsy, 
        sigma_delta, 
        bunch_length, 
        sr_inputs.equilibrium_epsx, 
        sr_inputs.equilibrium_epsy, 
        sr_inputs.equilibrium_sigma_delta, 
        sr_inputs.tau_x, 
        sr_inputs.tau_y, 
        sr_inputs.tau_z, 
        dt=time_step,
    )
    # --------------------------------------------------------------------
    # Compute new emittances at next turn with same time step and compare
    # First with NagaitsevIBS
    epsx, epsy, sigd, bl = NIBS.emittance_evolution(
        epsx=geom_epsx,
        epsy=geom_epsy,
        sigma_delta=sigma_delta,
        bunch_length=bunch_length,
        sr_equilibrium_epsx=sr_inputs.equilibrium_epsx,
        sr_equilibrium_epsy=sr_inputs.equilibrium_epsy,
        sr_equilibrium_sigma_delta=sr_inputs.equilibrium_sigma_delta,
        sr_tau_x=sr_inputs.tau_x,
        sr_tau_y=sr_inputs.tau_y,
        sr_tau_z=sr_inputs.tau_z,
        dt=time_step,
    )
    assert np.isclose(epsx, ref_epsx, atol=0)
    assert np.isclose(epsy, ref_epsy, atol=0)
    assert np.isclose(sigd, ref_sigma_delta, atol=0)
    # --------------------------------------------------------------------
    # And then with BjorkenMtingwaIBS
    epsx, epsy, sigd, bl = BMIBS.emittance_evolution(
        epsx=geom_epsx,
        epsy=geom_epsy,
        sigma_delta=sigma_delta,
        bunch_length=bunch_length,
        sr_equilibrium_epsx=sr_inputs.equilibrium_epsx,
        sr_equilibrium_epsy=sr_inputs.equilibrium_epsy,
        sr_equilibrium_sigma_delta=sr_inputs.equilibrium_sigma_delta,
        sr_tau_x=sr_inputs.tau_x,
        sr_tau_y=sr_inputs.tau_y,
        sr_tau_z=sr_inputs.tau_z,
        dt=time_step,
    )
    assert np.isclose(epsx, ref_epsx, atol=0)
    assert np.isclose(epsy, ref_epsy, atol=0)
    assert np.isclose(sigd, ref_sigma_delta, atol=0)
    # --------------------------------------------------------------------
    # We can do the same again with 1s as time step
    ref_epsx, ref_epsy, ref_sigma_delta = MIBS.emit_evol_with_SR(
        geom_epsx, 
        geom_epsy, 
        sigma_delta, 
        bunch_length, 
        sr_inputs.equilibrium_epsx, 
        sr_inputs.equilibrium_epsy, 
        sr_inputs.equilibrium_sigma_delta, 
        sr_inputs.tau_x, 
        sr_inputs.tau_y, 
        sr_inputs.tau_z, 
        dt=1,
    )
    # --------------------------------------------------------------------
    # Compute new emittances at next turn with same time step and compare
    # First with NagaitsevIBS
    epsx, epsy, sigd, bl = NIBS.emittance_evolution(
        epsx=geom_epsx,
        epsy=geom_epsy,
        sigma_delta=sigma_delta,
        bunch_length=bunch_length,
        sr_equilibrium_epsx=sr_inputs.equilibrium_epsx,
        sr_equilibrium_epsy=sr_inputs.equilibrium_epsy,
        sr_equilibrium_sigma_delta=sr_inputs.equilibrium_sigma_delta,
        sr_tau_x=sr_inputs.tau_x,
        sr_tau_y=sr_inputs.tau_y,
        sr_tau_z=sr_inputs.tau_z,
        dt=1,
    )
    assert np.isclose(epsx, ref_epsx, atol=0)
    assert np.isclose(epsy, ref_epsy, atol=0)
    assert np.isclose(sigd, ref_sigma_delta, atol=0)
    # --------------------------------------------------------------------
    # And then with BjorkenMtingwaIBS
    epsx, epsy, sigd, bl = BMIBS.emittance_evolution(
        epsx=geom_epsx,
        epsy=geom_epsy,
        sigma_delta=sigma_delta,
        bunch_length=bunch_length,
        sr_equilibrium_epsx=sr_inputs.equilibrium_epsx,
        sr_equilibrium_epsy=sr_inputs.equilibrium_epsy,
        sr_equilibrium_sigma_delta=sr_inputs.equilibrium_sigma_delta,
        sr_tau_x=sr_inputs.tau_x,
        sr_tau_y=sr_inputs.tau_y,
        sr_tau_z=sr_inputs.tau_z,
        dt=1,
    )
    assert np.isclose(epsx, ref_epsx, atol=0)
    assert np.isclose(epsy, ref_epsy, atol=0)
    assert np.isclose(sigd, ref_sigma_delta, atol=0)

# ----- Helpers ----- #


def _get_sr_inputs_from_line(line: xt.Line) -> _SynchrotronRadiationInputs:
    """As shown in the FAQ. Returns geometric equilibrium emittances."""
    # Set the radiation mode to 'mean' and call twiss with 'eneloss_and_damping' (see Xsuite user guide)
    line.configure_radiation(model="mean")
    twiss: xt.TwissTable = line.twiss(eneloss_and_damping=True)

    # The damping times, in [s] are provided as:
    sr_tau_x, sr_tau_y, sr_tau_z = twiss["damping_constants_s"]

    # For the geometric equilibrium emittances
    sr_equilibrium_gepsx = twiss["eq_gemitt_x"]
    sr_equilibrium_gepsy = twiss["eq_gemitt_y"]

    # We will need to store the equilibrium longitudinal emittance too for later
    sr_eq_zeta = twiss["eq_gemitt_zeta"]  # geometric

    # The equilibrium momentum spread is not directly provided but can be obtained via
    # a method of the twiss result, using the equilibrium emittances obtained above.
    # Make sure to use the right type based on the one you retrieved previously
    beam_sizes = twiss.get_beam_covariance(
        gemitt_x=sr_equilibrium_gepsx, gemitt_y=sr_equilibrium_gepsy, gemitt_zeta=sr_eq_zeta
    )

    # The value we want corresponds to the 'sigma_pzeta' key in this result, since in Xsuite it is equivalent
    # to 'sigma_delta' (see Xsuite physics guide, Eq 1.14 and 1.23). Take it at the location of the particles:
    sr_equilibrium_sigma_delta = beam_sizes["sigma_pzeta"][0]  # 0 for end / start of line

    # Send back the result
    return _SynchrotronRadiationInputs(
        sr_equilibrium_gepsx, sr_equilibrium_gepsy, sr_equilibrium_sigma_delta, sr_tau_x, sr_tau_y, sr_tau_z
    )


def _get_sr_inputs_from_madx(madx: Madx) -> _SynchrotronRadiationInputs:
    """As shown in the FAQ. Returns geometric equilibrium emittances."""
    # Set radiation flag on the beam, then call the 'emit' command with DELTAP=0, which will
    # update the beam with equilibrium values directly
    madx.input("beam, radiate;")
    madx.input("emit, deltap=0;")

    # The geometric transverse equilibrium emittances, in [m], are provided as:
    madx.input("eq_ex = beam->ex;")
    madx.input("eq_ey = beam->ey;")
    sr_equilibrium_gepsx = madx.globals["eq_ex"]
    sr_equilibrium_gepsy = madx.globals["eq_ey"]

    # The equilibrium momentum spread is not directly provided but can be obtained from
    # the relative energy spread using the relativistic beta as:
    madx.input("eq_sigd = beam->sige / beam->beta / beam->beta;")
    sr_equilibrium_sigma_delta = madx.globals["eq_sigd"]

    # We will need to get from the active beam: particle energy, energy loss per
    # turn (in [GeV]) and the revolution frequency (in [MHz])
    madx.input("E0 = beam->energy;")
    madx.input("U0 = beam->U0;")
    madx.input("frev = beam->freq0;")
    E0 = madx.globals["E0"] * 1e9
    U0 = madx.globals["U0"] * 1e9
    frev = madx.globals["frev"] * 1e6

    # We will need the synchrotron radiation integrals to determine the
    # damping partition numbers (see https://arxiv.org/pdf/1507.02213.pdf)
    madx.command.twiss(chrom=True)  # chrom to trigger their calculation
    I2 = madx.table.summ.synch_2[0]
    I4 = madx.table.summ.synch_4[0]
    jx = 1 - I4 / I2  # horizontal damping partition number
    jz = 2 + I4 / I2  # longitudinal damping partition number

    # This is enough to compute the damping times (see https://arxiv.org/pdf/1507.02213.pdf)
    sr_tau_x = 2 * E0 * frev / (jx * U0)
    sr_tau_y = 2 * E0 * frev / U0
    sr_tau_z = 2 * E0 * frev / (jz * U0)

    # Send back the result
    return _SynchrotronRadiationInputs(
        sr_equilibrium_gepsx, sr_equilibrium_gepsy, sr_equilibrium_sigma_delta, sr_tau_x, sr_tau_y, sr_tau_z
    )
