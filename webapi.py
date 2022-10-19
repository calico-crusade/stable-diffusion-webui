import json
from flask import Flask, jsonify, request
import os
import threading
import time
import importlib
import signal
import threading

from PIL import Image

from fastapi.middleware.gzip import GZipMiddleware

from modules.paths import script_path

from modules import devices, sd_samplers
import modules.codeformer_model as codeformer
import modules.extras
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.shared as shared
import modules.txt2img

import modules.ui
from modules import devices
from modules import modelloader
from modules.paths import script_path
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

import base64
from io import BytesIO

app = Flask(__name__)
queue_lock = threading.Lock()

def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)
        return res
    return f

def imageToBytes(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    bytes = base64.b64encode(buffered.getvalue())
    return bytes.decode('UTF-8')

@app.route('/txt2img', methods=['POST'])
def txt2img():
    data = json.loads(request.data)

    txt2img_prompt = data['prompt']
    txt2img_negative_prompt = data['negativePrompt']
    txt2img_prompt_style = 'None'
    txt2img_prompt_style2 = 'None'
    steps = data['steps']
    sampler_index = 0
    restore_faces = False
    tiling = False
    batch_count = data['batchCount']
    batch_size = data['batchSize']
    cfg_scale = data['cfgScale']
    seed = data['seed']
    subseed = -1
    subseed_strength = 0
    seed_resize_from_h = 0
    seed_resize_from_w = 0
    seed_checkbox = False
    height = data['height']
    width = data['width']
    enable_hr = False
    denoising_strength = 0.7
    firstphase_width = 0
    firstphase_height = 0
    args = [0, False, False, '', 'Seed', '', 'Nothing', '', True, False, False, None, '', '']

    res = modules.txt2img.txt2img(
        txt2img_prompt,
        txt2img_negative_prompt,
        txt2img_prompt_style,
        txt2img_prompt_style2,
        steps,
        sampler_index,
        restore_faces,
        tiling,
        batch_count,
        batch_size,
        cfg_scale,
        seed,
        subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
        height,
        width,
        enable_hr,
        denoising_strength,
        firstphase_width,
        firstphase_height,
        *args
    )

    (images, info, html) = res

    base64Images = [imageToBytes(x) for x in images]

    return jsonify({
        'info': info,
        'images': base64Images,
        'html': html
    })

@app.route('/img2img', methods=['POST'])
def img2img():
    data = json.loads(request.data)

    image_base64 = data['image']
    bytes_from_bas64 = base64.b64decode(image_base64)
    img = Image.open(BytesIO(bytes_from_bas64))

    mode = 0
    img2img_prompt = data['prompt']
    img2img_negative_prompt = data['negativePrompt']
    img2img_prompt_style = 'None'
    img2img_prompt_style2 = 'None'
    init_img = img
    init_img_with_mask = None
    init_img_inpaint = None
    init_mask_inpaint = None
    mask_mode = 0
    steps = data['steps']
    sampler_index = 0
    mask_blur = 0
    inpainting_fill = None
    restore_faces = False
    tiling = False
    batch_count = data['batchCount']
    batch_size = data['batchSize']
    cfg_scale = data['cfgScale']
    denoising_strength = data['denoiseStrength']
    seed = data['seed']
    subseed = -1
    subseed_strength = 0
    seed_resize_from_h = 0
    seed_resize_from_w = 0
    seed_checkbox = False
    height = data['height']
    width = data['width']
    resize_mode = 0
    inpaint_full_res = False
    inpaint_full_res_padding = 0
    inpainting_mask_invert = 0
    img2img_batch_input_dir = None
    img2img_batch_output_dir = None
    args = [0, False, False, '', 'Seed', '', 'Nothing', '', True, False, False, None, '', '']

    res = modules.img2img.img2img(
        mode,
        img2img_prompt,
        img2img_negative_prompt,
        img2img_prompt_style,
        img2img_prompt_style2,
        init_img,
        init_img_with_mask,
        init_img_inpaint,
        init_mask_inpaint,
        mask_mode,
        steps,
        sampler_index,
        mask_blur,
        inpainting_fill,
        restore_faces,
        tiling,
        batch_count,
        batch_size,
        cfg_scale,
        denoising_strength,
        seed,
        subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
        height,
        width,
        resize_mode,
        inpaint_full_res,
        inpaint_full_res_padding,
        inpainting_mask_invert,
        img2img_batch_input_dir,
        img2img_batch_output_dir,
        *args
    )

    (images, info, html) = res

    base64Images = [imageToBytes(x) for x in images]

    return jsonify({
        'info': info,
        'images': base64Images,
        'html': html
    })

@app.route('/embeddings', methods=['GET'])
def embeddings():
    items = os.listdir('./embeddings')
    return jsonify({
        'embeddings': items
    })


def init():
    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    shared.face_restorers.append(modules.face_restoration.FaceRestoration())
    modelloader.load_upscalers()

    modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

    shared.sd_model = modules.sd_models.load_model()
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))
    shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.hypernetworks.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))
    shared.opts.onchange("sd_hypernetwork_strength", modules.hypernetworks.hypernetwork.apply_strength)

    app.run()

if __name__ == "__main__":
    init()