# autonomous_car_driving
## Introduction
The goal of this project was implement the [NVIDIA paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
using behavioral cloning to make an End-to-End self-driving car with the Udacity [simulator](https://github.com/udacity/self-driving-car-sim) 

## Track 1
*   The Udacity self-driving car simulator provides two tracks, the track 1 is the easy one. So we drive the car manually to get data(images from the cameras and steering angles)from 3 to 4 laps.

*   The front cameras images will serve as input to our model and it will
try to predict the respective steering angle, so it will be the output of the model

*   With this data, i trained the pilotNet for 20 epochs,
what gave me a good result but the car was not able to do the final right turn because of the data imbalance of the training set. In this track there are much more left turns than right turns so almost all the training examples have a negative output(steering angle), thus the probability of the model output a negative value(left turn) is much higher than a positive value(right turn).

*   To correct this data imbalance problem i used data augmentation and this
is the result:

## Track 2
* This track is much harder compared to the track 1, because it has a lot of sharp turns and shaddows in the road. Here i modified the pilotNet replacing the elu activation by a relu and using batchNormalization in each layer.
Check out the result:


* Now the challenge is to make the car stay in only one lane. I've trained the same model with the data of only one lane but i had to help manually the car to make three turns in the road. Until now the throttle was calculated using the steering angle and the max speed, some turns that the car is not able to do is because it does not desacelerate enought. With that in mind
i modified the model to it predict not only the steering angle but also the throttle.
