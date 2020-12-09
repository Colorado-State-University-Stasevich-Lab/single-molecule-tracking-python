# single-molecule-tracking-python
#### This folder contains data for testing the code

Input data:     
1. 3D video (.tif) 
2. Particle tracking file (.csv). The tracking file minimally requires columns 
  'TRACK_ID', 'POSITION_T', 'POSITION_Y', and 'POSITION_X'    

Output data: files produced from SMT python code

## Installing packages

To run the jupyter notebook, make sure you have teh following programs installed on your computer: 

#### Python3 
https://www.python.org/downloads/mac-osx/

#### Anaconda for python3 
https://www.anaconda.com/products/individual

#### Upgrade pip 
`pip install --upgrade pip`

#### Napari 
`pip install napari[all]`

#### PyQt5 
`pip install PyQt5`

#### SciKit 
`pip install -U numpy scipy scikit-learn`

## Running the code 
In a command terminal, open jupyter notebook: 
`jupyter notebook` 

When redirected to a the online interface, upload the python code: 
* Download tracking code from this GitHub repository 
* Upload it into the jupyter notebook web gui
* *optional* Download additional input data files 
