# single-molecule-tracking-python
#### This code is for using Python to analyze single molecule tracking data derived from TrackMate 

#### Development repository : in-progress code and test files
#### Code repository : Final release 

Input data:     
1. 3D video (.tif) 
2. Particle tracking file (.csv). The tracking file minimally requires columns 
  'TRACK_ID', 'POSITION_T', 'POSITION_Y', and 'POSITION_X'    

Output data: files produced from SMT python code

## Installing packages
To run the jupyter notebook, make sure you have teh following programs installed on your computer.

### Mac OS : 

#### Python3 
https://www.python.org/downloads/mac-osx/

Make sure you have python3 installed: 

`python --version`

`Python 3.8.5`

#### Anaconda for python3 
`bash ~/Downloads/Anaconda3-2020.02-MacOSX-x86_64.sh` 

https://www.anaconda.com/products/individual

#### Upgrade pip 
`pip install --upgrade pip`

#### Napari 
`pip install napari[all]`

#### PyQt5 
`pip install PyQt5`

#### SciKit 
`pip install -U numpy scipy scikit-learn`


### Another way to install packages using GUI

#### Download and install the latest version of Anaconda
    https://www.anaconda.com/products/individual

#### Install Napari through Anaconda-Navigator
    go to "environments"
    add "conda-forge" into channels
    select "Not installed"
    type "Napari" in the search field
    add Napari
    
#### Install PyQt5
    Napari should run without installing PyQt5 (I think), but it will give a warning message about qt library version
    you can remove this warning message by installing PyQt5 from Terminal (updating qt from Anaconda-Navigator did not work for me)
      pip install PyQt5
    Do not try to update all libraries from terminal (conda update --all). I got many problems after updating all libraries.

## Running the code 
In a command terminal, open jupyter notebook: 
`jupyter notebook` 

When redirected to a the online interface, upload the python code: 
* Download tracking code from this GitHub repository 
* Upload it into the jupyter notebook web gui
* *optional:* Download additional input data files from this GitHub repository if you wish to use them to test the code
