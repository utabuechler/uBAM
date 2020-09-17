#Components

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

