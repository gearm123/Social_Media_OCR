import cv2
import torch
import numpy as np
try:
    import craft_utils
    import imgproc
except ModuleNotFoundError:
    craft_utils = None
    imgproc = None

def detect_text(image, craft_net):
    if craft_utils is None or imgproc is None:
        raise ModuleNotFoundError(
            "CRAFT helper modules are not available. Install/configure CRAFT-pytorch to use detect_text()."
        )
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image,
        1280,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=1.5
    )

    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        y, _ = craft_net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    boxes, _ = craft_utils.getDetBoxes(
        score_text,
        score_link,
        text_threshold=0.55,
        link_threshold=0.3,
        low_text=0.3,
        poly=False
    )

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    return boxes