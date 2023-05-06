model_name="R-50_RGB_60k"

python ./third_party/detectron2/tools/deploy/export_model.py \
    --sample-image ./1344x1344.jpg \
    --config-file exports/PercepTreeV1/$model_name/$model_name.yaml \
    --export-method tracing \
    --format onnx \
    --output exports/PercepTreeV1/$model_name \
    MODEL.WEIGHTS exports/PercepTreeV1/$model_name/$model_name.pkl \
    MODEL.DEVICE cuda