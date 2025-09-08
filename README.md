# QUEEN-BEE  
**Q-U Event-by-Event Nested sampling for Bayesian EVPA Evolution** 

Version: `3.2`  
Authors: Swati Ravi, Mason Ng, Herman Marshall  

---

## Overview  
QUEEN-BEE is a Bayesian framework for analyzing X-ray polarization data, with a particular focus on Imaging X-ray Polarimetry Explorer (IXPE) observations. It implements **nested sampling** (via [`bilby`](https://git.ligo.org/lscsoft/bilby)) to infer polarization properties directly from event data in **Stokes (q,u)** space, accounting for rotation of the polarization angle (EVPA).  

The package provides three baseline models for polarization:  

- **Unpolarized** (`scout_unpolar`) – null hypothesis, q = u = 0  
- **Constant** (`scout_const`) – constant polarization degree and angle  
- **Rotating** (`scout_rot`) – constant polarization degree with a linearly rotating EVPA over time  

QUEEN-BEE is intended for scientific use only and comes with **ABSOLUTELY NO WARRANTY**.  

---

## Features
- Modular likelihood functions:
  - Constant polarization (`scout_const`)
  - Rotating polarization (`scout_rot`)
  - Unpolarized reference model (`scout_unpolar`)

---

## Installation  

Clone the repository and install dependencies:  

```bash
git clone https://github.com/swati-ravi/QUEEN-BEE
cd QUEEN-BEE
pip install -r requirements.txt
```

## Requirements
Dependencies are listed in `requirements.txt`. Core packages include:
- Python 3.10+
- `numpy`
- `scipy`
- `astropy`
- `matplotlib`
- `bilby`
- `dynesty`
- `corner`
- `mpmath`

---

## Usage  

QUEEN-BEE can be run as a **command-line tool** or **imported as a Python module**.  

### Command-line usage  

```bash
python queen-bee.py [model] [options]
```

**Arguments:**  
- `model`: One of `unpolarized`, `constant`, `rotating`  
- `-v, --verbose`: Increase output verbosity (enables debug logging)  
- `--version`: Print version and contact information  

**Examples:**  
```bash
python queen-bee.py unpolarized
python queen-bee.py constant -v
python queen-bee.py rotating
```

---

### Python API usage  

```python
from queen-bee import scout_const, scout_rot, scout_unpolar

# Run constant polarization inference
scout_const()

# Run EVPA rotation inference
scout_rot()
```

---
## Function Specifications  

### `scout_unpolar()`  
Bayesian evidence computation for an unpolarized model (q = u = 0).  
- **Outputs:** log-evidence ± uncertainty  

### `scout_const()`  
Inference of constant polarization Stokes parameters (q, u).  
- **Outputs:** posterior samples, median values, 90% credible intervals, corner plot  

### `scout_rot()`  
Inference of Stokes parameters (q, u) and rotation rate of EVPA.  
- **Outputs:** posterior samples, median values, 90% credible intervals, corner plot  

---

## Data Input  

QUEEN-BEE requires:  
- **IXPE event level 2 files** (`evt2.fits`)  
- **Modulation factor files** (`.fits` in `modfact/`)  
- Source and background region definitions (arcsec) 
- Source centroid (CCD pixel coordinates) 
- (optional) for faint sources, background rejected files (`rej.fits') using the standard prescription

All input paths and energy selections (`emin`, `emax`) are configured inside the main script (`queen-bee.py`).  

---

## Output
- Posterior samples and results are stored in an output directory named:
  - `<source>_scout_unpolar`
  - `<source>_scout_const`
  - `<source>_scout_rot`
- Corner plots (`.png`) of posterior distributions are generated and displayed.
- Bilby result files (`.json`, `.h5`) in model-specific output directories    
- Console printout of posterior median values and 90% credible intervals  
- Log-evidence values for model comparison 

---

## License
This is free research software provided as-is for scientific use.
You are welcome to redistribute it under certain conditions.
Licensed under the MIT License (see LICENSE file for details).

---

## Disclaimer
QUEEN-BEE comes with ABSOLUTELY NO WARRANTY.  

---

## Contact
For questions, please contact

[swatir@mit.edu](mailto:swatir@mit.edu)

---

## Citation  

If you use QUEEN-BEE in your research, please cite:  

Ravi, S., Ng, M., Marshall, H., and Gnarini A., (2025). *What’s the Buzz About GX 13+1? Constraining Coronal Geometry with QUEEN-BEE: A Bayesian Nested Sampling Framework for X-ray Polarization Rotation Analysis.* 