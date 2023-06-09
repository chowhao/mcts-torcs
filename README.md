This code is for paper "Driving Maneuvers Prediction based Autonomous Driving Control by Deep Monte Carlo Tree Search", which has been accepted by IEEE Transactions on Vehicular Technology: https://ieeexplore.ieee.org/document/9082903

This code is running on Ubuntu 18.04

Train for Udacity simulator 
```
cd deep-MCTS
python coach_for_udacity.py
```
Train for Torcs simulator
```
cd deep-MCTS
python coach_for_Torcs.py
```
How to set up environments
```
pip install numpy==1.16.4
python-socketio==4.3.1
eventlet==0.25.1
Flask==1.1.1
Pillow==6.1.0
opencv-python==4.1.0.25
```
Without GPU
```text
conda install tensorflow==1.13.1
```
GPU
```text
conda install tensorflow-gpu==1.13.1
```