from uuid import uuid4
from io import BytesIO
import base64
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import PIL.Image
import pickle
from dnnlib import tflib
from style_loader import Dataset
from Matrix import MulLayer
import torchvision.utils as vutils

from style_transcoder import Encoder4, Decoder4

app = Flask(__name__, static_url_path='')
CORS(app)

def encode_img(image: any):
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=100)
    img_bytes = base64.b64encode(buffered.getvalue())
    img_str = 'data:image/jpeg;base64,' + img_bytes.decode('ascii')
    return img_str

def load_networks(path_or_url):
    stream = open(path_or_url, 'rb')
    session = tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')
        input_shape = Gs.input_shape[1:]
    return session, G, D, Gs, input_shape

session, _, _, Gs, input_shape = load_networks('models/art_1.pkl')

@app.route('/gen-art', methods=['POST'])
def gen_art():
    artist = request.json.get('artist')
    genre = request.json.get('genre')
    style = request.json.get('style')
    seed = request.json.get('seed')
    scale = request.json.get('scale')
    truncation = request.json.get('truncation')

    batch_size = 1
    l1 = np.zeros((1, 167))
    l1[0][artist] = 1
    l1[0][genre] = 1
    l1[0][style] = 1
    all_seeds = [seed] * batch_size
    all_z = np.stack(
        [np.random.RandomState(seed).randn(*input_shape) for seed in all_seeds])
    all_w = Gs.components.mapping.run(session=session, in_arrays=(
        scale * all_z, np.tile(l1, (batch_size, 1))))
    if truncation != 1:
        w_avg = Gs.get_var('dlatent_avg')
        all_w = w_avg + (all_w - w_avg) * truncation

    args = {
        'output_transform': {'func': tflib.convert_images_to_uint8, 'nchw_to_nhwc': True},
        'randomize_noise': True,
        'minibatch_size': 1
    }
    all_images = Gs.components.synthesis.run(session=session, in_arrays=(all_w,), **args)
    image = PIL.Image.fromarray(np.median(all_images, axis=0).astype(np.uint8))
    id = uuid4()
    image.save(f'art_{id}.png')
    return jsonify([
        encode_img(image),
        [f'art_{id}.png'],
    ])

content_v = torch.Tensor(1, 3, 256, 256)
style_v = torch.Tensor(1, 3, 256, 256)
vgg = Encoder4()
dec = Decoder4()
matrix = MulLayer('r41')
vgg.load_state_dict(torch.load(map_location=torch.device('cpu'), f='models/vgg_r41.pth'))
dec.load_state_dict(torch.load(map_location=torch.device('cpu'), f='models/dec_r41.pth'))
matrix.load_state_dict(torch.load(map_location=torch.device('cpu'), f='models/r41.pth'))


@app.route('/fusion', methods=['POST'])
def fusion():
    data = request.get_data()
    try:
        upload = PIL.Image.open(BytesIO(data)).convert('RGB')
    except Exception:
        return 'Not an image', 400
    file_name = request.headers.get('filename')
    gen_image = PIL.Image.open(file_name).convert('RGB')

    content_dataset = Dataset(upload, 256, 256, test=True)
    content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=1)
    style_dataset = Dataset(gen_image, 256, 256, test=True)
    style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1)

    for _, (content, _) in enumerate(content_loader):
        content_v.resize_(content.size()).copy_(content)
        for _, (style, _) in enumerate(style_loader):
            style_v.resize_(style.size()).copy_(style)

            with torch.no_grad():
                s_f = vgg(style_v)
                c_f = vgg(content_v)

                feature, transmatrix = matrix(c_f['r41'], s_f['r41'])
                transfer = dec(feature)

            transfer = transfer.clamp(0, 1)
            grid = vutils.make_grid(transfer, **{'normalize': True,
                                                 'scale_each': True, 'nrow': 1})
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            image = PIL.Image.fromarray(ndarr)
            image.save(f'fusion_{file_name}')
            return jsonify([
                encode_img(image),
                [f'fusion_{file_name}'],
            ])
