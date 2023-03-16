import onnxruntime as ort


def load_onnx(model_path, provider):
    return ort.InferenceSession(model_path, providers=[provider])


def run_onnx(onnx_model, x: dict):
    return onnx_model.run(None, x)
