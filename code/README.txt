There is no particular installation guideline, just need to install the libraries dependencies. 
But there are some files that are excluded due to its size.

1. It needs shape_predictor_68_face_landmarks.dat model to predict the facial landmark. 
The size is too big so it's not included, but it can be easily download online (through google search)

2. Only the top performing deep learning models are available. 
Total size of models checkpoints and history is over 110GB on the gPU server, so impossible to give all.
The model available are only vgg_custom_net_v19 epoch 130, AlexNet_optimized_v37 epoch 110. 
Transfer learning using VGG19 is not included because of the file size >70MB
It can be downloaded here https://drive.google.com/open?id=1pWFPzSCfLcH0MEL3_MV658IXrqKxS9Fn 
please extract the zip in the same folder location of the ipynb and py codes.

3. Facial images are not included due to the huge size. 
It can be downloaded from the internet (UTKFace), but have not been preprocessed. The preprocessing takes 10 hours