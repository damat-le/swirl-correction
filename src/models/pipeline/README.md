This is a 2-stage pipeline trained end2end:

1. Swirl detection: a UNet learns to predict the portion of the image where the swirl has been applied. It outputs a mask.
2. Swirl correction: a second custom UNet learns to correct the swirled image. The learning phase is driven by the swirl detection module since in the loss function we use the mask predicted from the previous step to focus on the relevant region of the image only. 

The best evaluation score achieved with this approach is however around 0.29.

The issue is probably in the swirl correction module (currently I used the sequential unet). On the other hand the swirl detection module seems to work pretty well.