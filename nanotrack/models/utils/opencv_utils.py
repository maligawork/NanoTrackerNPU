import cv2


def load_opencv(model_path, backend, target):
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)
    return net


def run_opencv(net, x, output_names):
    net.setInput(x)
    return net.forward(output_names)
