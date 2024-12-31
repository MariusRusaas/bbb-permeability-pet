
![BBBPS_3D_v0](https://github.com/user-attachments/assets/6573f4b4-d84a-4312-adc8-13dd4b446dfc)

## Quantitative PET imaging and modeling of molecular blood-brain barrier permeability
This repository contains the code and sample time-activity curve data for quantifying the blood-brain barrier (BBB) permeability of molecular radiotracers with high-temporal resolution dynamic positron emission tomography (PET).

## Usage
- Tested on Windows 10/11 and Python 3.9 to 3.11
```
# Download the Github repository
git clone https://github.com/kjch03/bbb-permeability-pet.git

# Navigate to the downloaded repository folder
cd bbb-permeability-pet

# Create a virtual environment
python -m venv venv-bbb

# ====================================
# Activate the virtual environment
# ====================================
# For Windows
venv-bbb/Scripts/activate

# For MacOS
source venv-bbb/bin/activate
# ====================================

# Install required packages
pip install -r requirements.txt

# Run the sample script
python run_aath_fitting.py
```

## Citation
We would appreciate citing our work if you use our method:
> Chung, K. J. et al. Quantitative PET imaging and modeling of molecular blood-brain barrier permeability. *medRxiv* (2024) doi:10.1101/2024.07.26.24311027.

## Contact
Please contact Kevin Chung (kjychung@ucdavis.edu) with any questions or comments.
