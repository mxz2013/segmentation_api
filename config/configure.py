# TODO: use proper model registration and yml instead of py file
VALID_MODEL_NAMES = {"mask_rcnn", "google/deeplabv3_mobilenet_v2_1.0_513"}

# allow segmentation class to be extended. If retrain model, server needs to be redeployed with new DEFAULT_TARGETS
# Inference code can accept user defined targets without redeployed the server so that each client has the possibility to extend target classes
DEFAULT_TARGETS = [8, 17]

# threshold to for the binary mask, user can input his own threshold from client as well.
DEFAULT_THRESHOLD = 0.5
