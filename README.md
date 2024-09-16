# SplitFedZip

Table of Contents

1. Overview
2. Features
3. Installation
4. Configuration
5. Usage
6. Contact

# Overview
SplitFedZip is the first rate-distortion based compression scheme desgined for splitFed networks. It aims to improce the communication efficiency in SplitFed Learning. This approach addresses key communication challenges like high latency, bandwidth constraints, and synchronization overheads in SplitFed learning while maintaining or enhancing model performance.

SplitFedZip was tested on medical image segmentation datasets, which achieved significant communication bitrate reduction with minimal impact on global model accuracy.

# Features

1. Learned Compression: SplitFedZip uses advanced codecs to compress feature maps and gradients at the split points.
2. Efficient Communication: It decreases communication overhead, hence addressing bandwidth and latency challenges common in federated contexts.
3. Flexible Architecture: Combines the benefits of Federated Learning (FL) and Split Learning (SL), balancing computational loads between the client and server.
4. Privacy Preservation: There is no need to share local data between clients and the server, therefore privacy is secured.


# Installation
Clone the repository:

git clone https://anonymous.4open.science/r/SplitFedZip.git 

cd SplitFedZip


Install required packages:

pip install -r requirements.txt


# Usage
SplitFedZip is designed with 4 compression schemes as (_NC scheme_) (No Compression), (_F scheme_) (Compressing only features), (_G scheme_)  (Compressing only gradients) and (_FG scheme_) (Compressing both features and gradients). The relevant codes for these 4 schemes are included in the /textit{baseline_splitFed_NC}, /textit{F scheme}, /textit{G scheme} and the /textit{FG scheme}.

To train the model on medical image segmentation data:


Use the runner.sh to start training. 




