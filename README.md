# YMDF model

YMDF stands for **Y**oung **M** **D**warf **F**lare. It consists of two modules: `flarefinder` and `model`.

<!-- MarkdownTOC -->

- [Credits](#credits)
- [Description](#description)
    - [Flarefinder](#flarefinder)
    - [Model](#model)
- [Installing](#installing)
    - [From PyPI](#from-pypi)
    - [From sources](#from-sources)
        - [Building a wheel](#building-a-wheel)
- [Tests](#tests)
- [References](#references)

<!-- /MarkdownTOC -->

## Credits

The module 'flarefinder' is based on [AltaiPony](https://github.com/ekaterinailin/AltaiPony) by [Ekaterina Ilin](https://ekaterinailin.github.io/). This module can work with white light curves in Kepler and TESS, and potentially new surveys as Plato, along with light curves produced by integrating the time-resolved spectra from the surveys working in other wavelength ranges, such as FUV COS-HST. Please, credit this module as Mamonova et al, 2025 ([1](#ref1)) (DOI (pending): 10.1051/0004-6361/202554614) and Ilin et al. (2021) ([2](#ref2)).

The module 'model' is an original work. Please, credit the module 'model' as Mamonova et al. 2026, in prep. ([3](#ref3))

## Description

### Flarefinder

De-trends light curves and searches for flare events. Injects synthetic flares and attempts their recovery to quantify the effects of de-trending and noise on flare energy, and to establish the energy-dependent recovery probability for each flare candidate. Utilizes `lightkurve` as the core analysis tool, along with `pandas`, `numpy`, `pytest`, `astropy`, and additional packages

### Model

Produces a time-resolved spectrum of flare activity in a Young M Dwarf. Most suitable for studying exoplanet atmospheres orbiting such stars. The reference star is the M dwarf AU Microscopii (M0.5-1, 24 Myr old, $M_\star = 0.5M_\odot$). The spectral part is based on radiative-hydrodynamic (RHD) synthetic models of the M dwarf stellar atmospheres during the flare event (Kowalski et al., 2022 ([4](#ref4)) , 2024 ([5](#ref5)), 2025 ([6](#ref6)) ). The flare frequency distributions (FFDs) used in the model are based on results in Mamonova et al. ([1](#ref1)), and the module offers both single and broken power law relations in FFDs.

## Installing

<!--
### From PyPI

``` sh
$ pip install phab-ymdf
```
-->

### From sources

``` sh
$ cd /path/to/phab-ymdf/
$ pip install .
```

Add an `-e` argument, if you'd like to automatically update your locally installed package by pulling from the repository or/and if you intend to modify the sources:

``` sh
$ pip install -e .
```

#### Building a wheel

You can also build a wheel and distribute/install that instead:

``` sh
$ cd /path/to/phab-ymdf/
$ python -m build
$ pip install ./dist/phab_ymdf-*.whl
```

## Tests

To run tests:

``` sh
$ pip install pytest

$ python -m pytest ./src/phab/tests/*[^_*].py
$ python -m pytest ./src/phab/tests/model.py
$ python -m pytest ./src/phab/tests/some.py -k "test_some_thing"
```

## References

1. <span id="ref1"></span> Mamonova E., Shan Y., Kowalski A. F., Werner S. C., 2025, "Flare frequency in M dwarfs belonging to young moving groups", arXiv, arXiv:2506.04465. https://doi.org:10.48550/arXiv.2506.04465

2. <span id="ref2"></span>  Ekaterina Ilin, Sarah J. Schmidt, Katja Poppenhäger, James R. A. Davenport, Martti H. Kristiansen, Mark Omohundro (2021). "Flares in Open Clusters with K2. II. Pleiades, Hyades, Praesepe, Ruprecht 147, and M67" Astronomy & Astrophysics, Volume 645, id.A42, 25 pp. https://doi.org/10.1051/0004-6361/202039198

3. <span id="ref3"></span>  Mamonova et al. 2026, "Paving the way for studying exoplanetary atmosphere around M dwarfs – Validating a new flare activity model", in prep.

4. <span id="ref4"></span>  Kowalski A. F., 2022, "Near-Ultraviolet Continuum Modeling of the 1985 April 12 Great Flare of AD Leo". Frontiers in Astronomy and Space Sciences 9. https://doi.org/10.3389/fspas.2022.1034458

5. <span id="ref5"></span>  Kowalski A. F., Allred J. C., Carlsson M., 2024, "Time-dependent Stellar Flare Models of Deep Atmospheric Heating". The Astrophysical Journal 969, https://doi.org/10.3847/1538-4357/ad4148

6. <span id="ref6"></span>  Kowalski A. F., Osten R. A., Notsu Y., Tristan I. I., Segura A., Maehara H., Namekata K., et al., 2025, "Rising Near-ultraviolet Spectra in Stellar Megaflares". The Astrophysical Journal 978 81. https://doi.org/10.3847/1538-4357/ad9395
