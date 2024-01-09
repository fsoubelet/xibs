# XIBS

<p>
  <!-- Github Release -->
  <a href="https://github.com/fsoubelet/xibs/releases">
    <img alt="Github Release" src="https://img.shields.io/github/v/release/fsoubelet/xibs?color=orange&label=Release&logo=Github">
  </a>

  <!-- PyPi Version -->
  <a href="https://pypi.org/project/xibs">
    <img alt="PyPI Version" src="https://img.shields.io/pypi/v/xibs?label=PyPI&logo=PyPI">
  </a>

  <!-- Github Actions Build -->
  <a href="https://github.com/fsoubelet/xibs/actions?query=workflow%3A%22Cron+Testing%22">
    <img alt="Github Actions" src="https://github.com/fsoubelet/xibs/workflows/Tests/badge.svg">
  </a>

  <!-- General DOI -->
  <a href="https://zenodo.org/badge/latestdoi/10044627.">
    <img alt="DOI" src="https://zenodo.org/records/10044627..svg">
  </a>
</p>

This repository contains the source for `xibs`, a prototype for Intra-Beam Scattering (IBS) modelling and computing to be later integrated into [Xsuite](https://github.com/xsuite).

> [!NOTE]
> This started as a fork of M. Zampetakis' work on a simple but physics-accurate
> [IBS implementation](https://github.com/MichZampetakis/IBS_for_Xsuite). The new
> package's code is quite different from the original but is benchmarked against
> it as well as other tools to ensure the validity of results.

See the [documentation](https://fsoubelet.github.io/xibs/) for details.

## Installing

Installation is easily done in your environment via `pip`:

```bash
python -m pip install xibs
```

## License

This project is licensed under the `Apache-2.0 License` - see the [LICENSE](LICENSE) file for details.
