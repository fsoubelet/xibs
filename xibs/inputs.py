"""
.. _xibs-inputs:

Input Data Structures
---------------------

Module with container dataclasses to encompass expected necessary inputs .
Various parts of ``xibs`` perform calculations based on beam and optics parameters.

The former is encompassed in a `BeamParameters` dataclass which is initiated from a generated `xpart.Particles` object.
The latter is encompassed in an `OpticsParameters` dataclass which is initiated from the result of a ``TWISS`` call on
the line (if in ``xsuite``) or the sequence (if in ``MAD-X``).
"""
from __future__ import annotations  # important for sphinx to alias ArrayLike

from dataclasses import InitVar, dataclass, field
from logging import getLogger
from typing import Optional, Union

import numpy as np

from numpy.typing import ArrayLike
from scipy.constants import c

LOGGER = getLogger(__name__)

# ----- Dataclasses for inputs to Nagaitsev class ----- #


@dataclass
class BeamParameters:
    r"""
    .. versionadded:: 0.2.0

    Container dataclass for necessary beam parameters. It is initiated from
    the `xpart.Particles` object to track in your line with ``xsuite``.

    Args:
        particles (xpart.Particles): the generated particles to be tracked or used
            in the line. This is an init-only parameter used for instanciation and
            it will not be kept in the instance's attributes.

    Attributes:
        n_part (int): number of simulated particles.
        particle_charge (int): elementary particle charge, in # of Coulomb charges
            (for instance 1 for electron or proton).
        particle_mass_eV (float): particle mass in [eV].
        total_energy_eV (float): total energy of the simulated particles in [eV].
        gamma_rel (float): relativistic gamma of the simulated particles.
        beta_rel (float): relativistic beta of the simulated particles.
        particle_classical_radius_m (float): the particles' classical radius in [m].
    """

    # ----- To be provided at initialization ----- #
    # Almost all is derived from these, but it is not kept!
    particles: InitVar["xpart.Particles"]  # noqa: F821
    # ----- Below are attributes derived from the Particles object ----- #
    # The following are Npart, Ncharg, E_rest, EnTot, gammar, betar and c_rad in Michalis's code
    n_part: int = field(init=False)
    particle_charge: int = field(init=False)
    particle_mass_eV: float = field(init=False)
    total_energy_eV: float = field(init=False)
    gamma_rel: float = field(init=False)
    beta_rel: float = field(init=False)
    particle_classical_radius_m: float = field(init=False)

    def __post_init__(self, particles: "xpart.Particles"):  # noqa: F821
        # Attributes derived from the Particles object
        LOGGER.debug("Initializing BeamParameters from Particles object")
        self.n_part = particles.weight[0] * particles.gamma0.shape[0]
        self.particle_charge = particles.q0
        self.particle_mass_eV = particles.mass0
        self.total_energy_eV = np.sqrt(particles.p0c[0] ** 2 + particles.mass0**2)
        self.gamma_rel = particles.gamma0[0]
        self.beta_rel = particles.beta0[0]
        self.particle_classical_radius_m = particles.get_classical_particle_radius0()

    @classmethod
    def from_madx(cls, madx: "cpymad.madx.Madx") -> BeamParameters:  # noqa: F821
        r"""
        .. versionadded:: 0.3.0

        Constructor to return a `BeamParameters` object from a `~cpymad.madx.Madx` object.
        This is a convenience method for the user, which essentially queries the relevant
        information from the current sequence's beam.

        .. warning::
            This method will query parameters from the `~cpymad.madx.Madx` object. It will
            get parameters from the current active sequence's beam, and use these to create
            an `xpart.Particles` object from which to instanciate the `BeamParameters` object.
            Note that the `xpart` package is required to use this method.

        Args:
            madx (cpymad.madx.Madx): a `~cpymad.madx.Madx` instance to created an `OpticsParameters`
                object from.

        Returns:
            A `BeamParameters` object.
        """
        import xpart as xp

        LOGGER.debug("Running TWISS for active sequence")
        madx.command.twiss()  # want the table to determine the sequence name and access its beam
        seq_name = madx.table.twiss.summary.sequence  # will give us the active sequence

        LOGGER.debug("Getting relevant information from current sequence's beam")
        gamma = madx.sequence[seq_name].beam.gamma  # relativistic gamma
        q0 = madx.sequence[seq_name].beam.charge  # electrical particle charge in units of [qp]
        mass0 = madx.sequence[seq_name].beam.mass * 1e9  # rest mass | in [GeV] in MAD-X but we want [eV]
        npart = madx.sequence[seq_name].beam.npart  # number of particles

        LOGGER.debug("Initializing BeamParameters from determined parameters")
        particle = xp.Particles(q0=q0, mass0=mass0, gamma0=gamma)
        result = cls(particle)
        result.n_part = int(npart)  # very important to adjust this!
        return result

    @classmethod
    def from_line(cls, line: "xtrack.Line", n_part: int) -> OpticsParameters:  # noqa: F821
        r"""
        .. versionadded:: 0.3.0

        Constructor to return a `BeamParameters` object from an `~xtrack.Line` object.
        This is a convenience method for the user, which will instantiate the class
        from the line's reference particle.

        Args:
            line (xtrack.Line): an `xtrack.Line`.
            n_part (int): number of particles to in the bunch. This is a mandatory
                argument as it is not possible to infer it from `line.particle_ref`.

        Returns:
            A `BeamParameters` object.
        """
        LOGGER.debug("Initializing BeamParameters from the line's reference particle")
        result = cls(line.particle_ref)
        result.n_part = int(n_part)  # very important to adjust this!
        return result


@dataclass
class OpticsParameters:
    r"""
    .. versionadded:: 0.2.0

    Container dataclass for necessary optics parameters. It is initiated from the results
    of a `.twiss()` command with an `xtrack.Line`, or result of a ``TWISS`` call in
    ``MAD-X`` as a dataframe (as given by ``cpymad`` by default).

    Args:
        twiss (Union["xtrack.twiss.TwissTable", pd.DataFrame]): the resulting `TwissTable`
            of a `.twiss()` call on the `xtrack.Line`, **or** the ``TWISS`` table of the
            ``MAD-X`` sequence as a `pandas.DataFrame`. Lowercase keys are expected. If
            initiating from a ``MAD-X`` twiss, the next two arguments are required. This
            is an init-only parameter used for instanciation and it will not be kept in
            the instance's attributes.
        _slipfactor (Optional[float]): the slip factor for the machine. Only required if
            the ``twiss`` argument is a ``MAD-X`` twiss dataframe.
        _frev_hz (Optional[float]): the revolution frequency for the machine in [Hz].
            Only required if the ``twiss`` argument is a ``MAD-X`` twiss dataframe.


    Attributes:
        s (ArrayLike): longitudinal positions of the machine elements in [m].
        circumference (float): machine circumference in [m].
        slip_factor (float): slip factor of the machine.
        revolution_frequency (float): revolution frequency of the machine in [Hz].
        betx (ArrayLike): horizontal beta functions in [m].
        bety (ArrayLike): vertical beta functions in [m].
        alfx (ArrayLike): horizontal alpha functions.
        alfy (ArrayLike): vertical alpha functions.
        dx (ArrayLike): horizontal dispersion functions in [m].
        dy (ArrayLike): vertical dispersion functions in [m].
        dpx (ArrayLike): horizontal dispersion of px (d px / d delta).
        dpy (ArrayLike): horizontal dispersion of py (d px / d delta).
    """

    # ----- To be provided at initialization ----- #
    twiss: InitVar[  # Almost all is derived from there, this is not kept!
        Union["xtrack.twiss.TwissTable", "pandas.DataFrame"]  # noqa: F821
    ]
    _slipfactor: InitVar[Optional[float]] = None
    _frev_hz: InitVar[Optional[float]] = None
    # ----- Below are attributes derived from the twiss table ----- #
    s: ArrayLike = field(init=False)
    circumference: float = field(init=False)
    slip_factor: float = field(init=False)
    revolution_frequency: float = field(init=False)
    betx: ArrayLike = field(init=False)
    bety: ArrayLike = field(init=False)
    alfx: ArrayLike = field(init=False)
    alfy: ArrayLike = field(init=False)
    # The following four are eta_x, eta_y, eta_dx and eta_dy in Michalis's code and in the old litterature
    dx: ArrayLike = field(init=False)
    dy: ArrayLike = field(init=False)
    dpx: ArrayLike = field(init=False)
    dpy: ArrayLike = field(init=False)
    # The following is private and just for us, information necessary in the
    _is_centered: bool = field(init=False)

    def __post_init__(
        self,
        twiss: Union["xtrack.twiss.TwissTable", "pandas.DataFrame"],  # noqa: F821
        # The following are only needed if we instanciate from MAD-X twiss
        _slipfactor: Optional[float] = None,
        _frev_hz: Optional[float] = None,
    ):
        # Attributes derived from the TwissTable
        self.s = np.array(twiss.s)
        self.circumference = self.s[-1]
        self.betx = np.array(twiss.betx)
        self.bety = np.array(twiss.bety)
        self.alfx = np.array(twiss.alfx)
        self.alfy = np.array(twiss.alfy)
        self.dx = np.array(twiss.dx)
        self.dy = np.array(twiss.dy)
        self.dpx = np.array(twiss.dpx)
        self.dpy = np.array(twiss.dpy)
        try:  # assume we have been given an xtrack.TwissTable
            self.slip_factor = twiss["slip_factor"]
            self.revolution_frequency = twiss.beta0 * c / self.circumference
            self._is_centered = False  # never the case in xsuite
            LOGGER.debug("Initialized OpticsParameters from TwissTable object")
        except KeyError:  # then it is a MAD-X twiss dataframe and we need these 2 provided
            self.slip_factor = _slipfactor
            self.revolution_frequency = _frev_hz
            self._is_centered = _is_twiss_centered(twiss)
            LOGGER.debug("Initialized OpticsParameters from MAD-X Twiss dataframe")

    @classmethod
    def from_madx(cls, madx: "cpymad.madx.Madx", **kwargs) -> OpticsParameters:  # noqa: F821
        r"""
        .. versionadded:: 0.3.0

        Constructor to return an `OpticsParameters` object from a `~cpymad.madx.Madx` object.
        This is a convenience method for the user, which essentially follows the steps described
        in the :ref:`FAQ <xibs-faq-optics-params-from-madx>`.

        .. warning::
            This method will query parameters from the `~cpymad.madx.Madx` object. It will
            get parameters from the current active sequence, and make a ``TWISS`` call for
            said sequence.

        Args:
            madx (cpymad.madx.Madx): a `~cpymad.madx.Madx` instance to created an `OpticsParameters`
                object from.
            **kwargs: any keyword argument will be transmitted to the `madx.twiss` call.
                The default `centre` argument is ``true`` (recommended) but it can be overriden.

        Returns:
            An `OpticsParameters` object.
        """
        centre = kwargs.pop("centre", True)  # by default centered, can be overriden
        LOGGER.debug("Running TWISS for active sequence")
        twiss = madx.twiss(centre=centre, **kwargs).dframe()  # this way we are sure to get a centered twiss
        seq_name = madx.table.twiss.summary.sequence  # will give us the active sequence

        LOGGER.debug("Getting slip factor and revolution frequency")
        frev_hz = madx.sequence[seq_name].beam.freq0 * 1e6  # freq0 in MHz in MADX & we want [Hz]
        gamma_rel = madx.sequence[seq_name].beam.gamma  # relativistic gamma
        gamma_tr = madx.table.summ.gammatr[0]  # transition gamma
        slipfactor = (1 / (gamma_tr**2)) - (1 / (gamma_rel**2))  # use the xsuite convention!

        cminus = madx.table.summ.dqmin[0]  # just to check coupling
        if not np.isclose(cminus, 0, atol=0, rtol=1e-4):  # there is some betatron coupling
            LOGGER.warning(
                f"There is betatron coupling in the machine (|Cminus| = {cminus:.3f}),"
                "which is not taken into account in analytical calculations."
            )

        LOGGER.debug("Initializing OpticsParameters from determined parameters")
        return cls(twiss, slipfactor, frev_hz)

    @classmethod
    def from_line(cls, line: "xtrack.Line", **kwargs) -> OpticsParameters:  # noqa: F821
        r"""
        .. versionadded:: 0.3.0

        Constructor to return an `OpticsParameters` object from an `~xtrack.Line` object.
        This is a convenience method for the user, which will perform a `.twiss` call on
        the provided line and return an `OpticsParameters` object.

        Args:
            line (xtrack.Line): an `xtrack.Line`.
            **kwargs: any keyword argument will be transmitted to the `line.twiss` call.
                The default `method` argument is ``4d`` but it can be overriden.

        Returns:
            An `OpticsParameters` object.
        """
        method = kwargs.pop("method", "4d")  # by default 4D, can be overriden
        LOGGER.debug("Running TWISS for on the line")
        twiss = line.twiss(method=method, **kwargs)

        if not np.isclose(twiss.c_minus, 0, atol=0, rtol=1e-4):  # there is some betatron coupling
            LOGGER.warning(
                f"There is betatron coupling in the machine (|Cminus| = {twiss.c_minus:.3f}),"
                "which is not taken into account in analytical calculations."
            )

        LOGGER.debug("Initializing OpticsParameters from determined parameters")
        return cls(twiss)


# ----- Helper functions ----- #


def _is_twiss_centered(twiss: "pandas.DataFrame") -> bool:  # noqa: F821
    r"""
    .. versionadded:: 0.3.0

    Determines if the ``TWISS`` was performed at the center of elements, in ``MAD-X``.
    If the `twiss` was obtained from ``xsuite`` it is never centered and this function
    is not necessary.

    .. tip::
        The check is performed as in the Fortran code of the ``IBS`` module in ``MAD-X``.
        We skip all rows in the table until we get to the first element with non-zero length,
        note its `s` position, then the `s` and `l` of the next element. We compare the
        :math:`\Delta s` to the length and conclude. If the two match, then the :math:`\Delta s`
        is exactly the length of the second element which means the `s` values are given at the
        end of elements, and therefore we are not centered. Otherwise, we are.

    Args:
        twiss (pd.DataFrame): a DataFrame of the twiss table, from ``MAD-X``.

    Returns:
        `True` if the TWISS was centered, `False` otherwise.
    """
    # Get to the first row with an actual element of non-zero length
    tw = twiss[twiss.l != 0]
    # Get the "first" value of s and l
    s0 = tw.s.to_numpy()[0]
    # Get the s variable at the next element
    l1 = tw.l.to_numpy()[1]
    s1 = tw.s.to_numpy()[1]
    # Compare s1 - s0 with the length of the second element, if it matches then the delta_s corresponds
    # to the length of the element and we are getting values at the exit of the elements: not centered
    return not np.isclose(s1 - s0, l1)
