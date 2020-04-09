# OpenCV-readNetFromTensorflow-sample
OpenCV 4.X系のreadNetFromTensorflow()の動作サンプルです。

# Requirement
* OpenCV 4.0.0(or later)

# Note
以下のリポジトリのモデルをreadNetFromTensorflow()用に変換して使用しています。

https://github.com/Kazuhito00/hand-detection-3class-MobilenetV2-SSDLite

# Reference
OpenCV公式の説明に沿って変換を行っています。

https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

変換はmodelディレクトリの中で以下のコマンドで実行できます。

```bash
python tf_text_graph_ssd.py --input frozen_inference_graph.pb --config ssd_mobilenet_v2_coco.config --output graph.pbtxt
```

# Usage
 
サンプルの実行方法は以下です。
 
```bash
python hsv_mask_extracter.py
```

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
OpenCV-readNetFromTensorflow-sample is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

# OpenCV License
OpenCVはBSDライセンスの元で配布されています。

本リポジトリでは以下を使用しています。
* tf_text_graph_ssd.py
* tf_text_graph_common.py
* ssd_mobilenet_v2_coco.config

https://github.com/opencv/opencv/blob/master/LICENSE
