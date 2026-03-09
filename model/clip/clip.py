
import hashlib
import os
import urllib
import warnings
from typing import Union, List, Tuple

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

if torch.__version__.split(".") < ["1", "7", "1"]:
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

__all__ = ["available_models", "load", "tokenize", "build_CLIP_from_openai_pretrained"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B-32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B-16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")
    return download_target


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    return list(_MODELS.keys())


def _resolve_model_path(name: str, download_root: str = None):
    if name in _MODELS:
        return _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    if os.path.isfile(name):
        return name
    raise RuntimeError(f"Model {name} not found; available models = {available_models()}")


def _load_state_dict_from_name_or_path(name: str, jit: bool = False, download_root: str = None):
    model_path = _resolve_model_path(name, download_root=download_root)
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
        model = None
        state_dict = torch.load(model_path, map_location="cpu")
    return model_path, (state_dict or model.state_dict())


def _infer_resolutions(state_dict: dict, image_size: Union[int, Tuple[int, int]], stride_size: int):
    if isinstance(image_size, int):
        image_h = image_w = image_size
    else:
        image_h, image_w = image_size

    vit = "visual.proj" in state_dict
    if vit:
        patch_size = state_dict["visual.conv1.weight"].shape[-1]
        h_resolution = int((image_h - patch_size) // stride_size + 1)
        w_resolution = int((image_w - patch_size) // stride_size + 1)
    else:
        h_resolution = max(1, image_h // 32)
        w_resolution = max(1, image_w // 32)
    return h_resolution, w_resolution


def build_CLIP_from_openai_pretrained(name: str, image_size: Union[int, Tuple[int, int]], stride_size: int,
                                      jit: bool = False, download_root: str = None):
    model_path, state_dict = _load_state_dict_from_name_or_path(name, jit=jit, download_root=download_root)
    h_resolution, w_resolution = _infer_resolutions(state_dict, image_size, stride_size)
    model = build_model(state_dict, h_resolution, w_resolution, stride_size)
    model_cfg = {
        "model_path": model_path,
        "h_resolution": h_resolution,
        "w_resolution": w_resolution,
        "stride_size": stride_size,
    }
    return model, model_cfg


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=False):
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        # default CLIP path keeps the original 224/224-ish spatial setup; PromptSG-specific loading should use build_CLIP_from_openai_pretrained
        vit = "visual.proj" in (state_dict or model.state_dict())
        if vit:
            patch_size = (state_dict or model.state_dict())["visual.conv1.weight"].shape[-1]
            grid_size = round(((state_dict or model.state_dict())["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            h_resolution = w_resolution = grid_size
            vision_stride_size = patch_size
        else:
            output_width = round(((state_dict or model.state_dict())["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            h_resolution = w_resolution = output_width
            vision_stride_size = 32

        model = build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution if hasattr(model.visual, "input_resolution") else 224)

    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
