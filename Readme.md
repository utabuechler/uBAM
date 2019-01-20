Here we collect all our code for the nature methods journal.

Our code has the following parts:

1. Preprocessing: 
    Input: [Videos]
    - Frame Extraction
    - Tracking
    Output: [*.jpg, detections]

2. Features:
    Input: [*.jpg, detections]
    - train the CNN-LSTM network
    - extract features
    Output: [model weights, features]

3. Behavior Analysis:
    a) Similarities
        Input: [detections, features]
        - query NN
        - density
        - tsne with images
        Output: [plots]

    b) Behavior
        Input: [detections, features, weak labels]
        - LDA
        - SVM
        Output: [Plots]

    c) Disease Magnification
        Input: [detections, features, weak labels]
        - train VAE
        - density estimation
        - SVM
        - inter-/extraploation
        Output: [plots]


Unusual requirements:

- GPUtil: https://github.com/anderskm/gputil



TO DO:
- finish human walking tsne magnification
- finish the code for figure 2C3 (our framework)
- similarity: finish the code for the LDA similarity plots (should work for human and rats?)
- similarity: showing tsne of frames (and sequences?)

- do we want the 'homepage' to be user specific? with inputs etc.?


Provide:
- features (only) of rat dataset?
- provide a few videos for running through the network?
        
