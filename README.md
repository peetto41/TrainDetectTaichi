# TrainDetectTaichi
Code for doing AI to check Tai Chi dance moves
## Build Setup
``` bash
# install env
pip install virtualenv
```
``` bash
# create a python environment named env or whatever.
venv env
```
``` bash
# activate env for Linux and macOS
source env/bin/activate 
```
``` bash
# activate env for Windows
.\env\Scripts\activate
```
``` bash
# install mediapipe
pip install mediapipe
```
``` bash
# install opencv
pip install opencv-python
```
``` bash
# install decord
pip install decord
```
``` bash
# install pandas
pip install pandas
```
``` bash
# install scikit-learn
pip install scikit-learn
```
``` bash
# install matplotlib
pip install matplotlib
```
``` bash
# install matplotlib
pip install matplotlib
```
## How to run code
``` bash
## Step 1 Put the desired video in the folder.
## Step 2 run code to capture video to keep in dataset
python cp.py
## step 3 run code to train dataset and export model used
python train.py
## step 4 run code to check result model
python detect.py
```
