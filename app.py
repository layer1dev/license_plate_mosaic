import cv2
import torch
# import matplotlib.pyplot as plt
# from PIL import Image
from PIL import Image as Img
import numpy as np
import os
# import tempfile
# from six import BytesIO
# from six.moves.urllib.request import urlopen

# from utils.torch_utils import select_device, smart_inference_mode

from flask import Flask, request
from flask import send_file
from flask_cors import CORS

import uuid
# import json

# import requests
import urllib.request

# SETUP
weights = "./weights/bestv2.pt"

# model load ? cpu
device = 'gpu'

# custom trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', weights)

app = Flask(__name__)
CORS(app)
host = '0.0.0.0'
port = 50003

if not os.path.isdir('./tmp'):
    os.makedirs('./tmp')


@app.route("/np_mosaic", methods=['GET', 'POST'])
def GET():
    image_url = request.args.get("images")

    path, ext = os.path.splitext(image_url)

    unique_filename = str(uuid.uuid4())

    save_path = "./tmp/"+unique_filename+ext

    urllib.request.urlretrieve(image_url, save_path)

    im = cv2.imread(save_path)

    pred = model(im)

    results = pred.pandas().xyxy[0][['name', 'xmin', 'ymin', 'xmax', 'ymax']]

    imgRGB = ""

    for i, k in enumerate(results.values):
        print(results.values[i])

        rate = 30

        pred_point = results.values[i]

        x = int(pred_point[1])
        y = int(pred_point[2])
        w = (int(pred_point[3]) - x)
        h = (int(pred_point[4]) - y)

        # 관심영역 지정
        mosaic = im[y:y+h, x:x+w]
        # 축소
        mosaic = cv2.resize(mosaic, (w//rate, h//rate))
        # 확대
        mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_AREA)
        # 원본 이미지에 적용
        im[y:y+h, x:x+w] = mosaic

        imgRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if type(imgRGB) is not str:
        image_pil = Img.fromarray(np.uint8(imgRGB)).convert("RGB")

        save_path = './tmp/_'+unique_filename+ext

        image_pil.save(save_path)

    return send_file(save_path, as_attachment=True)


if __name__ == '__main__':
    app.run(host=host, port=port)
