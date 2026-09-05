import torch
from PIL import Image

from kraken_monkey_patch import apply_kraken_monkey_patch

apply_kraken_monkey_patch()

from kraken.models import load_models
from kraken.configs import RecognitionInferenceConfig
from kraken.lib.dataset import ImageInputTransforms


MODEL = "/home/gnikoul/.local/share/htrmopo/a328a8d1-29fb-519a-bf81-3f56a512f080/ppocr_v6_tau090.safetensors"
IMAGE = "datasets/eparchos/training/images/000000.png"


model = load_models(MODEL)[0]

config = RecognitionInferenceConfig(
    accelerator="cuda",
    devices=1,
    batch_size=1,
    padding=0,
)

model.prepare_for_inference(config)

print("Kraken model loaded:", type(model).__name__)
print("Device:", next(model.parameters()).device)


image = Image.open(IMAGE)

batch, channels, height, width = model.input

transforms = ImageInputTransforms(
    batch,
    height,
    width,
    channels,
    (model._inf_config.padding, 0),
    False,
    dtype=model._m_dtype,
)

with model._fabric.init_tensor():
    ts_im = transforms(image)

seqs = ts_im.unsqueeze(0)

seq_lens = torch.tensor(
    [ts_im.shape[2]],
    device=seqs.device,
)

preds, olens = model._rec_predict(seqs, seq_lens)

for pred in preds:
    text = ''.join(x[0] for x in pred)
    print("TRANSCRIPTION:", text)