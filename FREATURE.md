https://github.com/tianyu0207/RTFM/issues/2

> Can you kindly explain the process you followed for generating the i3d features of the shanghai tech dataset so that we can follow the same for other datasets and videos as well?

Video frames from non-overlapping sliding windows (16 frames each) are passed through the I3D network; features
are extracted from the ‘Mix 5c’ network layer, that are then reshaped to 2048-D vectors.

> 1. How do I reshape that to 2048D? (Add features generated using RGB and flow -> 1024+1024?)
> 2. Also, the dimension of your .npy files for the Shanghai Tech dataset is (some k,10,2048). What does each dimension indicate here? The paper indicates that the proposed RTFM receives T*D feature matrix (2 dimension) for one video, so I didn't understand why the uploaded features were in 3 dimension
> 3. Are the features generated for the shanghai tech dataset are only using RGB frames and not optical flow images in the given onedrive link?

1. Hi Please use the I3d network with Resnet 50 as the backbone to extract features.
2. To be consistent with the previous works, we use 10-crop augmentation, hence, 10 represents each cropped frame and k represents the number of 16-frames clips.
3. The generated feature only uses the RGB features.

> In another issue in this repo, I found that we need to divide the video to 32 snippets that means for any given video I'll get 32*2048 features, so won't k be fixed as 32 and not variable as you mentioned in the second point? Sorry if I didn't understand it correctly.

The feature is first extracted based on every 16-frames using I3D. Therefore, k = total-frames/16. Then during training, we process each video into 32 segments using process_feat function in util.py. This is the same as the paper 'real-world anomaly detection in surveillance videos'.

https://github.com/tianyu0207/RTFM/issues/5

> I have doubt regarding the dimension required of the input discussed in the paper. It is said that the input is a T*2048 feature vector for a given video. And it also said that T is taken as 32 in implementation details.
> 
> Does this mean, for any given video, we need to divide it into 32 parts (no matter the no.of frames) and find 1*2048 vector for each part?

Yes. For a batch of videos, we evenly divide each video into 32 parts and form a Batch * 32 * 2048 feature vector for this batch.

https://github.com/tianyu0207/RTFM/issues/8

> Will you share the code of extracting I3D feature? Thanks!

You can simply use this repo https://github.com/Tushar-N/pytorch-resnet3d.

https://github.com/tianyu0207/RTFM/issues/30

> I see your features‘ shape is N * 10 * 2048，but mix_5c output shape is 1 * 1024 when input is 16 * 224 * 224

You can simply use this resnet50 I3D to extract the feature. https://github.com/Tushar-N/pytorch-resnet3d

This produces 2048 dimensional features. 

https://github.com/tianyu0207/RTFM/issues/33

> * Can you tell me why I3D-10 crops features of XD-Violence are not provided in this git?
> * Please can you show me the I3D-10 crops feature extraction code? Because I extracted only 2 dimensions, not 3 dimensions like yours.

For XD-VIolence, You can just use the I3D feature provided in https://roc-ng.github.io/XD-Violence/.

Or, you can simply use this resnet50 I3D to extract the feature. https://github.com/Tushar-N/pytorch-resnet3d.

The 10-crop I use is just the Pytorch official 10-crop function.

> what is the input that you fed into the backbone I3D-Res? Is it every frame you fed into it or every 16 frames(aka one clip)?

I fed for every 16 frames.

