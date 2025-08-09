This network learns a transformation of the coordinates of each pixel.

Given a grid of pixels (H,W,2), the network first learns a set of F features per pixel (H,W,F) through a convolutional block. 

Then, the original grid, augmented with this set of features through concatenation (H,W,2+F), is passed to a feed-forward block that predicts the residual coordinates of each pixel in the grid.

The original and residual coordinates are summed up together to obtain the final pixel coordinates.

However, at the moment the network is able to reconstruct the input image (swirled) but it is not able to invert the swirl. I'm using the patchwise loss function to make the network focus on the swirled part of the image but it does not work. I'm not sure what the problem is, probably a simple feed-forward block is not enough to learn the swirl transformation. Maybe the features extracted by the conv block are not enough to inform the ffblock on where to map each pixel. At the moment, each pixel is treated quite independently from the othes, thus an idea can be to introduce an attention mechanism so that the network can learn dependencies among pixels.