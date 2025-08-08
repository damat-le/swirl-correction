This network learns a transformation of the coordinates of each pixel.

Given a grid of pixels (H,W,2), the network first learns a set of F features per pixel (H,W,F) through a convolutional block. 

Then, the original grid, augmented with this set of features through concatenation (H,W,2+F), is passed to a feed-forward block that predicts the residual coordinates of each pixel in the grid.

The original and residual coordinates are summed up together to obtain the final pixel coordinates.

However, at the moment the network is able to reconstruct the input image but it is not able to invert the swirl.