# autonomous_car_driving
##Introduction
The goal of this project was implement the NVIDIA paper:
using behavioral cloning to make an End-to-End self-driving car with the Udacity simulator:

##Track 1
*   The Udacity self-driving car simulator provides two tracks, the track 1 is the easy one. So we drive the car manually to get data(images from the cameras and steering angles)from 3 to 4 laps.

*   The front cameras images will serve as input to our model and it will
try to predict the respective steering angle, so it will be the output of the model

*   With this data, i trained the pilotNet for 20 epochs,
what gave me a good result but the car was not able to do the final right turn because of the data imbalance of the training set. In this track there are much more left turns than right turns so almost all the training examples have a negative output(steering angle), thus the probability of the model output a negative value(left turn) is much higher than a positive value(right turn).

*   To correct this data imbalance problem i used data augmentation and this
is the result:

##Track 2
* 