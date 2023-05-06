model_name="R-50_RGB_60k"

python third_party/TensorRT/samples/python/detectron2/create_onnx.py \
    --exported_onnx exports/PercepTreeV1/$model_name/model.onnx \
    --onnx exports/PercepTreeV1/$model_name/model-exported.onnx \
    --det2_config exports/PercepTreeV1/$model_name/$model_name.yaml \
    --det2_weights exports/PercepTreeV1/$model_name/$model_name.pkl \
    --sample_image ./1344x1344.jpg