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

*   To correct this data imbalance problem i used data augmentation: x and y translation, horrizontal mirroing, random shadows and bright.
Check out the result by clicking in the image bellow:
<div style='text-align:center'>
[![Watch the video](http://i3.ytimg.com/vi/LC6WGWp_Yik/hqdefault.jpg)](https://www.youtube.com/watch?v=LC6WGWp_Yik&t=6s)
</div>
## Track 2
* This track is much harder compared to the track 1, because it has a lot of sharp turns and shaddows in the road. Here i modified the pilotNet replacing the elu activation by a relu and using batchNormalization in each layer. I trained this model with an one lap data and knowing that this track has several sharp turns i modified the throttle equation f
rom: throttle = 1.0 - steering_angle**2 - (speed/limit)**2
to:  throttle = np.clip(1.0 - abs(steering_angle) - (speed/limit)**2, -1.0, 1.0)

so the car will desacelerate more to make the turn and and i multiplied the predicted steering_angle by 1.5 to make the car turn more.
Check out the result by clicking in the image bellow:

[![Watch the video](http://i3.ytimg.com/vi/VCH0dpJ3Rh4/hqdefault.jpg)](https://www.youtube.com/watch?v=VCH0dpJ3Rh4)


## Future improviments
Somethings that might improve the model
* Try different architectures with Residual Learning or Recurrent neural networks
* Include the velocity and throttle in the network
* Get more data for the second track
* Try Reinforcement Learning
