import cv2
import numpy as np
import matplotlib.image as mpimg

def pre_process(image):
    #first the sky and the car could be remove from the image to make it smaller
    image = image[60:130, :, :]
    image = cv2.resize(image, (200,70), cv2.INTER_AREA)

    #Change color space to YUV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def select_image(center, left, right, steering_angle):
    #if we chose the the images from left or right camera
    #we have to compensate the steering angle by 0.2
    c = np.random.choice(3)
    if(c == 0):
        return mpimg.imread(center), steering_angle
    elif(c == 1):
        return mpimg.imread(left), steering_angle + 0.2
    else:
        return mpimg.imread(right), steering_angle - 0.2

def random_flip(image, steering_angle):
    #50% probability of flipping the image
    if(np.random.rand() < 0.5):
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle

    return image, steering_angle

def random_translate(image, steering_angle):
    #translate randomly the images in the range of
    #[-0.5*range,0.5*range]
    #As our images are more verticals the vertical range will be bigger
    range_x = 100
    range_y = 10
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)

    trans_matrix = np.float32([[1,0,trans_x], [0,1,trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_matrix, (width, height))
    steering_angle += trans_x*0.002

    return image, steering_angle

def random_shadow(image):
    height, width = image.shape[:2]
    #create a random line
    x1, y1 = width * np.random.rand(), 0
    x2, y2 = width * np.random.rand(), height
    xm, ym = np.mgrid[0:height, 0:width]

    #set all pixels bellow the line to 1 and 0 above the line
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_bright(image):
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augment_data(center, left, right, steering_angle):
    #choose between one of the three images
    image, steering = select_image(center, left, right, steering_angle)

    #flip the image
    image, steering = random_flip(image, steering)

    #randomly translate the image
    image, steering = random_translate(image, steering) 

    #randomly shadow the image
    image = random_shadow(image)

    #randomly bright the image
    image = random_bright(image)

    return image, steering


def batch_generator(image_paths, steering_angles, batch_size, is_training):
    #initialize the input and expected output to return
    X = np.empty([batch_size, 70, 200, 3])
    Y = np.empty(batch_size)

    while True:
        #Put image per image in the batch array X
        #So when count == batch_size we've got out batch complete
        count = 0

        #Go through all the image paths randomly
        for index in np.random.permutation(image_paths.shape[0]):
            #get the tree image paths(inputs) and the steering angle(expected output)
            center, left, right = image_paths[index]
            steering = steering_angles[index]

            #data augmentation
            if(is_training and np.random.rand() < 0.6):
                image, steering = augment_data(center, left, right, steering)

            else:
                image = mpimg.imread(center)
            
            X[count] = pre_process(image)
            Y[count] = steering

            count += 1
            if(count == batch_size):
                break
        
        yield X, Y
