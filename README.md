# SplitFedZip

Table of Contents

1. Introduction
2. Features
3. Configuration
4. Usage
5. Contact

# Overview
SplitFedZip is a novel method that employs learned compression to improve communication efficiency in Split-Federated (SplitFed) learning. This approach addresses key communication challenges like high latency, bandwidth constraints, and synchronization overheads in SplitFed learning while maintaining or enhancing model performance.

SplitFedZip was tested on medical image segmentation datasets, where it achieved significant communication bitrate reduction with minimal impact on global model accuracy.


# Features

1. Learned Compression: SplitFedZip incorporates advanced codecs to compress feature maps and gradients at split points.
2. Efficient Communication: It reduces communication overhead, mitigating bandwidth and latency issues typical in federated environments.
3. Flexible Architecture: Combines the strengths of Federated Learning (FL) and Split Learning (SL), balancing computational loads across client and server.
4. Privacy Preserving: No need to share local data between clients and the server, ensuring privacy is maintained.

