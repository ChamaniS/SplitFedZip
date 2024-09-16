# SplitFedZip

This repository is the Python implementation of the paper "SplitFedZIP: Learned Compression for Efficient Communication in Split-Federated Learning".

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
SplitFedZip is designed with 4 compression schemes as _NC scheme_ (No Compression), _F scheme_ (Compressing only features), _G scheme_ (Compressing only gradients) and _FG scheme_ (Compressing both features and gradients). The relevant codes for these 4 schemes are included in the _baseline_splitFed_NC_, _F scheme_, _G scheme_ and the _FG scheme_.

To train the model on medical image segmentation data:
- Go to the folder where you want to perform the compression.
- Add the location to data at: dataDir = LOCATION_TO_DATA
- Add the location for saving checkpoints, validated images, tested images, and output curves at: compressedF = LOCATION_TO_COMPRESSED_DATA_FOLDER
- Add the location to the output log at: output_file = LOCATION_TO_OUTPUT_FILE
- Change the value of the lagrangian multiplier at: lambda1 = _value_

Use the _runner.sh_ to start training. 

# Contact
Please contact the corresponding authors of the paper.

# Citation
If you find this project useful in your research, please cite our paper:

```bibtex
@article{Anonymous_2024,
  title={SplitFedZip: Efficient Communication in Split-Federated Learning via Learned Compression},
  author={Anonymous},
  journal={Journal Name},
  year={2024}
}


