## Perceptree to Tensorrt
adapted from https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples/python/detectron2

*This should be done on desktop, not jetson*

**download model**
- download model weights from https://github.com/norlab-ulaval/PercepTreeV1
- make sure the weights file matches the name of the model that you will use. Example: R-50_RGB_60k.pth for model R-50_RGB_60k

**Export yaml and pkl files from your model**
Edit lines 27 to 29 in "export_pkl_yaml.py" to reflect your models details:
```
model_name = "R-50_RGB_60k"
weights_path = "models/PercepTreeV1/R-50_RGB_60k.pth"
detectron2_config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
```
optional: edit lines 43-52 to customize model

*Run export_pkl_yaml.py*
``` python export_pkl_yaml.py ```

**Get third party repos and install requirements** 
```
mkdir third_party
cd third_party
git clone https://github.com/facebookresearch/detectron2.git
git clone https://github.com/NVIDIA/TensorRT.git

pip install -r ./TensorRT/samples/python/detectron2/requirements.txt
```

**Change these lines of detectron2**
```vim detectron2/tools/deploy/export_model.py```

change lines 165-167 from this:
```
aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)
```
to this:
```
aug = T.ResizeShortestEdge(
    [1344, 1344], 1344
)
```

**run conversion to onnx**
edit model_name variable in conversion1.sh and conversion2.sh to reflect the model you are using

```
cd ..
bash conversion1.sh
bash conversion2.sh
```

**onnx to tensorrt**
this can be run on desktop or jetson

example for R-50_RGB_60k
```
/usr/src/tensorrt/bin/trtexec --onnx=exports/PercepTreeV1/R-50_RGB_60k/model-exported.onnx --saveEngine=exports/PercepTreeV1/R-50_RGB_60k/engine.trt --useCudaGraph
```

**test tensorrt file**

example for R-50
```
mkdir predictions
python third_party/TensorRT/samples/python/detectron2/infer.py \
    --engine ./exports/PercepTreeV1/R-50_RGB_60k/engine.trt \
    --input tree_images \
    --det2_config ./exports/PercepTreeV1/R-50_RGB_60k/R-50_RGB_60k.yaml \
    --output ./predictions 
```