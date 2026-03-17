import os
import gradio as gr
import numpy as np
import random
import torch
import spaces
import base64
import json
from io import BytesIO
from PIL import Image, ImageDraw
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

MAX_SEED = np.iinfo(np.int32).max
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
        torch_dtype=dtype,
        device_map="cuda",
    ),
    torch_dtype=dtype,
).to(device)
try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

ADAPTER_SPECS = {
    "Object-Remover": {
        "repo": "prithivMLmods/QIE-2511-Object-Remover-Bbox-v3",
        "weights": "QIE-2511-Object-Remover-Bbox-v3-10000.safetensors",
        "adapter_name": "object-remover",
    },
    "Design-Adder": {
        "repo": "prithivMLmods/QIE-2511-Outfit-Design-Layout",
        "weights": "QIE-2511-Outfit-Design-Layout-3000.safetensors",
        "adapter_name": "design-adder",
    },
    "Object-Mover": {
        "repo": "prithivMLmods/QIE-2511-Object-Mover-Bbox",
        "weights": "QIE-2511-Object-Mover-Bbox-5000.safetensors",
        "adapter_name": "object-mover",
    },
}

loaded_adapters = set()
current_adapter = None

DEFAULT_PROMPTS = {
    "Object-Remover": "Remove the red highlighted object from the scene",
    "Design-Adder": "Add the design pattern inside the red highlighted bounding box area",
    "Object-Mover": "Move the object highlighted in the red box to the location indicated by the other red box in the scene",
}

EXAMPLE_IMAGES = ["examples/1.jpg", "examples/2.jpg", "examples/3.jpg"]


def b64_to_pil(b64_str):
    if not b64_str or not b64_str.startswith("data:image"):
        return None
    try:
        _, data = b64_str.split(",", 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def burn_boxes_onto_image(pil_image, boxes_json_str):
    if not pil_image:
        return pil_image
    try:
        boxes = json.loads(boxes_json_str) if boxes_json_str and boxes_json_str.strip() else []
    except Exception:
        boxes = []
    if not boxes:
        return pil_image
    img = pil_image.copy().convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    bw = max(3, w // 250)
    for b in boxes:
        x1 = int(b["x1"] * w)
        y1 = int(b["y1"] * h)
        x2 = int(b["x2"] * w)
        y2 = int(b["y2"] * h)
        lx, rx = min(x1, x2), max(x1, x2)
        ty, by_ = min(y1, y2), max(y1, y2)
        draw.rectangle([lx, ty, rx, by_], outline=(255, 0, 0), width=bw)
    return img


def make_thumb_b64(path, max_dim=220):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        thumb = img.copy()
        thumb.thumbnail((max_dim, max_dim), LANCZOS)
        buf = BytesIO()
        thumb.save(buf, format="JPEG", quality=65)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""


def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Encode error for {path}: {e}")
        return ""


def preload_example_thumbnails():
    results = []
    for path in EXAMPLE_IMAGES:
        try:
            img = Image.open(path).convert("RGB")
            thumb_b64 = make_thumb_b64(path, max_dim=280)
            results.append({
                "thumb": thumb_b64,
                "name": os.path.basename(path).rsplit(".", 1)[0].capitalize(),
                "size": f"{img.width}\u00d7{img.height}",
            })
        except Exception as e:
            print(f"Warning: Could not load example {path}: {e}")
            results.append(None)
    return results


def build_examples_html(thumbs):
    cards_html = ""
    for i, data in enumerate(thumbs):
        if data is None or not data.get("thumb"):
            continue
        cards_html += f'''<div class="example-card" data-example-idx="{i}" title="Click to load {data['name']}">
            <div class="example-img-wrap">
                <img src="{data['thumb']}" alt="{data['name']}" draggable="false" />
                <div class="example-overlay"></div>
            </div>
            <div class="example-info">
                <span class="example-label">{data['name']}</span>
                <span class="example-size">{data['size']}</span>
            </div>
        </div>'''
    return f'''<div class="examples-section">
        <div class="examples-title">Quick Examples</div>
        <div class="examples-grid">{cards_html}</div>
    </div>'''


def load_example_data(idx_str):
    try:
        idx = int(float(idx_str)) if idx_str and idx_str.strip() else -1
    except (ValueError, TypeError):
        idx = -1
    if idx < 0 or idx >= len(EXAMPLE_IMAGES):
        return json.dumps({"image": "", "status": "error"})
    path = EXAMPLE_IMAGES[idx]
    b64 = encode_full_image(path)
    if b64:
        return json.dumps({"image": b64, "name": os.path.basename(path), "status": "ok"})
    return json.dumps({"image": "", "status": "error"})


print("Building example thumbnails...")
example_thumbs = preload_example_thumbnails()
examples_html_block = build_examples_html(example_thumbs)
print(f"Built {len(EXAMPLE_IMAGES)} example cards.")


@spaces.GPU
def infer_bbox_task(
    b64_str,
    boxes_json,
    prompt,
    adapter_choice,
    seed=0,
    randomize_seed=True,
    guidance_scale=1.0,
    num_inference_steps=4,
    height=1024,
    width=1024,
):
    global loaded_adapters, current_adapter
    progress = gr.Progress(track_tqdm=True)
    if not prompt or prompt.strip() == "":
        raise gr.Error("\u26a0 Prompt cannot be empty.")
    spec = ADAPTER_SPECS.get(adapter_choice)
    if not spec:
        raise gr.Error(f"Unknown adapter: {adapter_choice}")
    adapter_name = spec["adapter_name"]
    source_image = b64_to_pil(b64_str)
    if source_image is None:
        raise gr.Error("Please upload an image first.")
    try:
        boxes = json.loads(boxes_json) if boxes_json and boxes_json.strip() else []
    except Exception:
        boxes = []
    if not boxes:
        raise gr.Error("Please draw at least one bounding box.")
    if adapter_choice == "Object-Mover" and len(boxes) != 2:
        raise gr.Error(
            f"\u26a0 Object Mover requires exactly 2 bounding boxes (Source + Target). You have {len(boxes)}."
        )
    if adapter_name not in loaded_adapters:
        progress(0.1, desc=f"Loading {adapter_choice} adapter...")
        pipe.load_lora_weights(spec["repo"], weight_name=spec["weights"], adapter_name=adapter_name)
        loaded_adapters.add(adapter_name)
    if current_adapter != adapter_name:
        progress(0.2, desc=f"Switching to {adapter_choice}...")
        pipe.set_adapters([adapter_name], adapter_weights=[1.0])
        current_adapter = adapter_name
    progress(0.3, desc="Burning red boxes onto image...")
    marked = burn_boxes_onto_image(source_image, boxes_json)
    progress(0.5, desc=f"Running {adapter_choice} inference...")
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        image=[marked],
        prompt=prompt,
        height=height if height != 0 else None,
        width=width if width != 0 else None,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
    ).images[0]
    return result, seed, marked


def update_dimensions_on_upload(b64_str):
    image = b64_to_pil(b64_str)
    if image is None:
        return 1024, 1024
    original_width, original_height = image.size
    if original_width > original_height:
        new_width = 1024
        new_height = int(new_width * original_height / original_width)
    else:
        new_height = 1024
        new_width = int(new_height * original_width / original_height)
    return (new_width // 8) * 8, (new_height // 8) * 8


css = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body,.gradio-container{
    background:#0f0f13!important;font-family:'Inter',system-ui,-apple-system,sans-serif!important;
    font-size:14px!important;color:#e4e4e7!important;min-height:100vh;
}
.dark body,.dark .gradio-container{background:#0f0f13!important;color:#e4e4e7!important}
footer{display:none!important}
.hidden-input{display:none!important;height:0!important;overflow:hidden!important;margin:0!important;padding:0!important}

#example-load-btn{
    position:absolute!important;left:-9999px!important;top:-9999px!important;
    width:1px!important;height:1px!important;opacity:0.01!important;
    pointer-events:none!important;overflow:hidden!important;
}
#gradio-run-btn{
    position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;
    opacity:0.01;pointer-events:none;overflow:hidden;
}

.toast-notification{
    position:fixed;top:24px;left:50%;transform:translateX(-50%) translateY(-120%);
    z-index:9999;padding:10px 24px;border-radius:10px;font-family:'Inter',sans-serif;
    font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px;
    box-shadow:0 8px 24px rgba(0,0,0,.5);
    transition:transform .35s cubic-bezier(.34,1.56,.64,1),opacity .35s ease;opacity:0;pointer-events:none;
}
.toast-notification.visible{transform:translateX(-50%) translateY(0);opacity:1;pointer-events:auto}
.toast-notification.error{background:linear-gradient(135deg,#dc2626,#b91c1c);color:#fff;border:1px solid rgba(255,255,255,.15)}
.toast-notification.warning{background:linear-gradient(135deg,#d97706,#b45309);color:#fff;border:1px solid rgba(255,255,255,.15)}
.toast-notification.info{background:linear-gradient(135deg,#C71585,#FF1493);color:#fff;border:1px solid rgba(255,255,255,.15)}
.toast-notification .toast-icon{font-size:16px;line-height:1}
.toast-notification .toast-text{line-height:1.3}

.app-shell{
    background:#18181b;border:1px solid #27272a;border-radius:16px;
    margin:12px auto;max-width:1400px;overflow:hidden;
    box-shadow:0 25px 50px -12px rgba(0,0,0,.6),0 0 0 1px rgba(255,255,255,.03);
}
.app-header{
    background:linear-gradient(135deg,#18181b,#1e1e24);border-bottom:1px solid #27272a;
    padding:14px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
}
.app-header-left{display:flex;align-items:center;gap:12px}
.app-logo{
    width:36px;height:36px;background:linear-gradient(135deg,#FF1493,#FF69B4,#FFB6C1);
    border-radius:10px;display:flex;align-items:center;justify-content:center;
    font-size:18px;font-weight:800;color:#fff;box-shadow:0 4px 12px rgba(255,20,147,.35);
}
.app-title{
    font-size:18px;font-weight:700;background:linear-gradient(135deg,#e4e4e7,#a1a1aa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-.3px;
}
.app-badge{
    font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;
    background:rgba(255,20,147,.15);color:#FF69B4;border:1px solid rgba(255,20,147,.25);letter-spacing:.3px;
}
.mode-switcher{
    display:flex;gap:4px;background:#09090b;border:1px solid #27272a;border-radius:10px;padding:3px;
}
.mode-btn{
    display:inline-flex;align-items:center;justify-content:center;gap:6px;
    padding:6px 16px;border:none;border-radius:8px;cursor:pointer;
    font-size:13px;font-weight:600;font-family:'Inter',sans-serif;
    color:#71717a;background:transparent;transition:all .2s ease;white-space:nowrap;
}
.mode-btn:hover{color:#a1a1aa;background:rgba(255,20,147,.08)}
.mode-btn.active{
    color:#fff!important;-webkit-text-fill-color:#fff!important;
    background:linear-gradient(135deg,#FF1493,#C71585)!important;
    box-shadow:0 2px 8px rgba(255,20,147,.35);
}
.mode-btn.active .mode-icon{color:#fff!important;-webkit-text-fill-color:#fff!important}
.mode-btn .mode-icon{font-size:15px;line-height:1}

.app-toolbar{
    background:#18181b;border-bottom:1px solid #27272a;padding:8px 16px;
    display:flex;gap:4px;align-items:center;flex-wrap:wrap;
}
.tb-sep{width:1px;height:28px;background:#27272a;margin:0 8px}
.modern-tb-btn{
    display:inline-flex;align-items:center;justify-content:center;gap:6px;
    min-width:32px;height:34px;background:transparent;border:1px solid transparent;
    border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;padding:0 12px;
    font-family:'Inter',sans-serif;color:#ffffff!important;transition:all .15s ease;
}
.modern-tb-btn:hover{background:rgba(255,20,147,.15);color:#ffffff!important;border-color:rgba(255,20,147,.3)}
.modern-tb-btn:active,.modern-tb-btn.active{background:rgba(255,20,147,.25);color:#ffffff!important;border-color:rgba(255,20,147,.45)}
.modern-tb-btn .tb-icon{font-size:15px;line-height:1;color:#ffffff!important}
.modern-tb-btn .tb-label{font-size:13px;color:#ffffff!important;font-weight:600}

.app-main-row{display:flex;gap:0;flex:1;overflow:hidden}
.app-main-left{flex:1;display:flex;flex-direction:column;min-width:0;border-right:1px solid #27272a}
.app-main-right{width:420px;display:flex;flex-direction:column;flex-shrink:0;background:#18181b}

#bbox-draw-wrap{position:relative;background:#09090b;margin:0;min-height:440px;overflow:hidden;cursor:crosshair}
#bbox-draw-wrap.drag-over{outline:2px solid #FF1493;outline-offset:-2px;background:rgba(255,20,147,.04)}
#bbox-draw-canvas{display:block;margin:0 auto}
#bbox-status{
    position:absolute;top:12px;left:12px;background:rgba(255,0,0,.9);color:#fff;
    padding:4px 12px;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:500;
    border-radius:6px;z-index:10;display:none;pointer-events:none;backdrop-filter:blur(8px);
}
#bbox-count{
    position:absolute;top:12px;right:12px;background:rgba(24,24,27,.9);color:#FF1493;
    padding:4px 12px;font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;
    border-radius:6px;border:1px solid rgba(255,20,147,.3);z-index:10;display:none;backdrop-filter:blur(8px);
}
#mover-box-hint{
    position:absolute;bottom:12px;left:50%;transform:translateX(-50%);
    background:rgba(24,24,27,.92);color:#FF69B4;padding:6px 16px;
    font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;
    border-radius:8px;border:1px solid rgba(255,20,147,.3);z-index:10;display:none;
    pointer-events:none;backdrop-filter:blur(8px);white-space:nowrap;
}

.upload-prompt-modern{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);z-index:20}
.upload-click-area{
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    cursor:pointer;padding:36px 52px;border:2px dashed #3f3f46;border-radius:16px;
    background:rgba(255,20,147,.03);transition:all .2s ease;gap:8px;
}
.upload-click-area:hover{background:rgba(255,20,147,.08);border-color:#FF1493;transform:scale(1.03)}
.upload-click-area:active{background:rgba(255,20,147,.12);transform:scale(.98)}
.upload-click-area svg{width:80px;height:80px}
.upload-main-text{color:#71717a;font-size:14px;font-weight:500;margin-top:4px}
.upload-sub-text{color:#52525b;font-size:12px}

.hint-bar{
    background:rgba(255,20,147,.06);border-top:1px solid #27272a;border-bottom:1px solid #27272a;
    padding:10px 20px;font-size:13px;color:#a1a1aa;line-height:1.7;
}
.hint-bar b{color:#FFB6C1;font-weight:600}
.hint-bar kbd{
    display:inline-block;padding:1px 6px;background:#27272a;border:1px solid #3f3f46;
    border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#a1a1aa;
}
.hint-bar .hint-mover-tag{
    display:inline-block;padding:1px 8px;background:rgba(255,20,147,.15);
    border:1px solid rgba(255,20,147,.3);border-radius:4px;font-size:11px;font-weight:600;color:#FF69B4;
}

.json-panel{
    background:#18181b;border-top:1px solid #27272a;display:flex;flex-direction:column;
    height:160px;max-height:160px;min-height:160px;
}
.json-panel-title{
    padding:8px 16px;font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;
    letter-spacing:.8px;border-bottom:1px solid #27272a;display:flex;align-items:center;gap:8px;flex-shrink:0;
}
.json-panel-title::before{
    content:'{ }';font-family:'JetBrains Mono',monospace;font-size:11px;color:#FF1493;
    background:rgba(255,20,147,.12);padding:2px 6px;border-radius:4px;
}
.json-panel-content{
    background:#09090b;margin:0;padding:12px 16px;font-family:'JetBrains Mono',monospace;
    font-size:12px;color:#a1a1aa;flex:1;overflow-y:auto;overflow-x:hidden;
    word-break:break-all;white-space:pre-wrap;line-height:1.6;
}
.json-panel-content::-webkit-scrollbar{width:8px}
.json-panel-content::-webkit-scrollbar-track{background:#09090b}
.json-panel-content::-webkit-scrollbar-thumb{background:#27272a;border-radius:4px}
.json-panel-content::-webkit-scrollbar-thumb:hover{background:#3f3f46}

.examples-section{background:#18181b;border-top:1px solid #27272a}
.examples-title{
    padding:8px 16px;font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;
    letter-spacing:.8px;border-bottom:1px solid #27272a;display:flex;align-items:center;gap:8px;
}
.examples-title::before{
    content:'\2b29';font-size:12px;color:#FF1493;background:rgba(255,20,147,.12);
    padding:2px 6px;border-radius:4px;
}
.examples-grid{display:flex;gap:10px;padding:12px 16px;overflow-x:auto}
.examples-grid::-webkit-scrollbar{height:6px}
.examples-grid::-webkit-scrollbar-track{background:#18181b}
.examples-grid::-webkit-scrollbar-thumb{background:#27272a;border-radius:3px}
.examples-grid::-webkit-scrollbar-thumb:hover{background:#3f3f46}
.example-card{
    flex:0 0 auto;width:150px;cursor:pointer;border:2px solid #27272a;border-radius:10px;
    overflow:hidden;transition:all .2s ease;background:#09090b;user-select:none;-webkit-user-select:none;position:relative;
}
.example-card:hover{
    border-color:#FF1493;transform:translateY(-3px);
    box-shadow:0 6px 20px rgba(255,20,147,.25),0 0 0 1px rgba(255,20,147,.1);
}
.example-card:active{transform:translateY(-1px) scale(.98);box-shadow:0 2px 8px rgba(255,20,147,.2)}
.example-card.example-loading{pointer-events:none;opacity:.7}
.example-card.example-loading .example-overlay{display:flex!important;background:rgba(9,9,11,.85)!important}
.example-card.example-loading .example-overlay::after{
    content:'';width:24px;height:24px;border:2px solid #27272a;border-top-color:#FF1493;
    border-radius:50%;animation:spin .7s linear infinite;
}
.example-card.example-active{
    border-color:#FF69B4!important;
    box-shadow:0 0 0 2px rgba(255,20,147,.3),0 4px 12px rgba(255,20,147,.2)!important;
}
.example-img-wrap{position:relative;overflow:hidden}
.example-img-wrap img{width:100%;height:95px;object-fit:cover;display:block;transition:transform .3s ease}
.example-card:hover .example-img-wrap img{transform:scale(1.06)}
.example-overlay{
    position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(9,9,11,.5);
    display:none;align-items:center;justify-content:center;transition:all .2s ease;
}
.example-info{
    padding:7px 10px;display:flex;align-items:center;justify-content:space-between;
    border-top:1px solid #27272a;gap:4px;
}
.example-label{font-size:12px;font-weight:600;color:#a1a1aa}
.example-size{
    font-family:'JetBrains Mono',monospace;font-size:10px;color:#52525b;
    background:#18181b;padding:1px 6px;border-radius:4px;border:1px solid #27272a;
}

.panel-card{border-bottom:1px solid #27272a}
.panel-card-title{
    padding:12px 20px;font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;
    letter-spacing:.8px;border-bottom:1px solid rgba(39,39,42,.6);
}
.panel-card-body{padding:16px 20px;display:flex;flex-direction:column;gap:8px}
.modern-label{font-size:13px;font-weight:500;color:#a1a1aa;margin-bottom:4px;display:block}
.modern-textarea{
    width:100%;background:#09090b;border:1px solid #27272a;border-radius:8px;
    padding:10px 14px;font-family:'Inter',sans-serif;font-size:14px;color:#e4e4e7;
    resize:vertical;outline:none;min-height:42px;transition:border-color .2s;
}
.modern-textarea:focus{border-color:#FF1493;box-shadow:0 0 0 3px rgba(255,20,147,.15)}
.modern-textarea::placeholder{color:#3f3f46}
.modern-textarea.error-flash{
    border-color:#FF1493!important;box-shadow:0 0 0 3px rgba(255,20,147,.25)!important;animation:shake .4s ease;
}
@keyframes shake{0%,100%{transform:translateX(0)}20%,60%{transform:translateX(-4px)}40%,80%{transform:translateX(4px)}}

.btn-run{
    display:flex;align-items:center;justify-content:center;gap:8px;width:100%;
    background:linear-gradient(135deg,#FF1493,#C71585);border:none;border-radius:10px;
    padding:12px 24px;cursor:pointer;font-size:15px;font-weight:600;font-family:'Inter',sans-serif;
    color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;transition:all .2s ease;letter-spacing:-.2px;
    box-shadow:0 4px 16px rgba(255,20,147,.3),inset 0 1px 0 rgba(255,255,255,.1);
}
.btn-run:hover{
    background:linear-gradient(135deg,#FF69B4,#FF1493);transform:translateY(-1px);
    box-shadow:0 6px 24px rgba(255,20,147,.45),inset 0 1px 0 rgba(255,255,255,.15);
}
.btn-run:active{transform:translateY(0);box-shadow:0 2px 8px rgba(255,20,147,.3)}
.btn-run svg{width:18px;height:18px;fill:#ffffff!important}
.btn-run svg path{fill:#ffffff!important}
#custom-run-btn,#custom-run-btn *,#custom-run-btn span,#custom-run-btn svg,
#custom-run-btn svg path,#run-btn-label,.btn-run,.btn-run *{
    color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;fill:#ffffff!important;
}
.btn-run.design-mode{background:linear-gradient(135deg,#FF1493,#DB7093)!important}
.btn-run.design-mode:hover{background:linear-gradient(135deg,#FF69B4,#FF1493)!important}
.btn-run.mover-mode{background:linear-gradient(135deg,#C71585,#FF1493)!important}
.btn-run.mover-mode:hover{background:linear-gradient(135deg,#FF69B4,#C71585)!important}

body:not(.dark) .btn-run,body:not(.dark) .btn-run *,body:not(.dark) #custom-run-btn,
body:not(.dark) #custom-run-btn *{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;fill:#ffffff!important}
.dark .btn-run,.dark .btn-run *,.dark #custom-run-btn,.dark #custom-run-btn *{
    color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;fill:#ffffff!important;
}
.gradio-container .btn-run,.gradio-container .btn-run *,.gradio-container #custom-run-btn,
.gradio-container #custom-run-btn *{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;fill:#ffffff!important}

.mode-btn.active,.mode-btn.active *,.mode-btn.active span,.mode-btn.active .mode-icon{
    color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;
}
body:not(.dark) .mode-btn.active,body:not(.dark) .mode-btn.active *{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}
.dark .mode-btn.active,.dark .mode-btn.active *{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}
.gradio-container .mode-btn.active,.gradio-container .mode-btn.active *{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}

body:not(.dark) .modern-tb-btn,body:not(.dark) .modern-tb-btn *{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}
.dark .modern-tb-btn,.dark .modern-tb-btn *{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}
.gradio-container .modern-tb-btn,.gradio-container .modern-tb-btn *{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}

.output-frame{border-bottom:1px solid #27272a;display:flex;flex-direction:column;position:relative}
.output-frame .out-title{
    padding:10px 20px;font-size:13px;font-weight:700;color:#ffffff!important;
    -webkit-text-fill-color:#ffffff!important;text-transform:uppercase;letter-spacing:.8px;
    border-bottom:1px solid rgba(39,39,42,.6);display:flex;align-items:center;justify-content:space-between;
}
.output-frame .out-title span{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}
.output-frame .out-body{
    flex:1;background:#09090b;display:flex;align-items:center;justify-content:center;
    overflow:hidden;min-height:180px;position:relative;
}
.output-frame .out-body img{max-width:100%;max-height:460px;image-rendering:auto}
.output-frame .out-placeholder{color:#3f3f46;font-size:13px;text-align:center;padding:20px}
.out-download-btn{
    display:none;align-items:center;justify-content:center;background:rgba(255,20,147,.1);
    border:1px solid rgba(255,20,147,.2);border-radius:6px;cursor:pointer;padding:3px 10px;
    font-size:11px;font-weight:500;color:#FFB6C1!important;gap:4px;height:24px;transition:all .15s;
}
.out-download-btn:hover{background:rgba(255,20,147,.2);border-color:rgba(255,20,147,.35);color:#ffffff!important}
.out-download-btn.visible{display:inline-flex}
.out-download-btn svg{width:12px;height:12px;fill:#FFB6C1}

.modern-loader{
    display:none;position:absolute;top:0;left:0;right:0;bottom:0;background:rgba(9,9,11,.92);
    z-index:15;flex-direction:column;align-items:center;justify-content:center;gap:16px;backdrop-filter:blur(4px);
}
.modern-loader.active{display:flex}
.modern-loader .loader-spinner{
    width:36px;height:36px;border:3px solid #27272a;border-top-color:#FF1493;
    border-radius:50%;animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.modern-loader .loader-text{font-size:13px;color:#a1a1aa;font-weight:500}
.loader-bar-track{width:200px;height:4px;background:#27272a;border-radius:2px;overflow:hidden}
.loader-bar-fill{
    height:100%;background:linear-gradient(90deg,#FF1493,#FF69B4,#FF1493);
    background-size:200% 100%;animation:shimmer 1.5s ease-in-out infinite;border-radius:2px;
}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

.settings-group{border:1px solid #27272a;border-radius:10px;margin:12px 16px;padding:0;overflow:hidden}
.settings-group-title{
    font-size:12px;font-weight:600;color:#71717a;text-transform:uppercase;letter-spacing:.8px;
    padding:10px 16px;border-bottom:1px solid #27272a;background:rgba(24,24,27,.5);
}
.settings-group-body{padding:14px 16px;display:flex;flex-direction:column;gap:12px}
.slider-row{display:flex;align-items:center;gap:10px;min-height:28px}
.slider-row label,.slider-row .dim-label{font-size:13px;font-weight:500;color:#a1a1aa;min-width:72px;flex-shrink:0}
.slider-row input[type="range"]{
    flex:1;-webkit-appearance:none;appearance:none;height:6px;background:#27272a;
    border-radius:3px;outline:none;min-width:0;
}
.slider-row input[type="range"]::-webkit-slider-thumb{
    -webkit-appearance:none;width:16px;height:16px;background:linear-gradient(135deg,#FF1493,#C71585);
    border-radius:50%;cursor:pointer;box-shadow:0 2px 6px rgba(255,20,147,.4);transition:transform .15s;
}
.slider-row input[type="range"]::-webkit-slider-thumb:hover{transform:scale(1.2)}
.slider-row input[type="range"]::-moz-range-thumb{
    width:16px;height:16px;background:linear-gradient(135deg,#FF1493,#C71585);
    border-radius:50%;cursor:pointer;border:none;box-shadow:0 2px 6px rgba(255,20,147,.4);
}
.slider-row .slider-val{
    min-width:52px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px;
    font-weight:500;padding:3px 8px;background:#09090b;border:1px solid #27272a;
    border-radius:6px;color:#a1a1aa;flex-shrink:0;
}
.checkbox-row{display:flex;align-items:center;gap:8px;font-size:13px;color:#a1a1aa}
.checkbox-row input[type="checkbox"]{accent-color:#FF1493;width:16px;height:16px;cursor:pointer}
.checkbox-row label{color:#a1a1aa;font-size:13px;cursor:pointer}

.app-statusbar{
    background:#18181b;border-top:1px solid #27272a;padding:6px 20px;
    display:flex;gap:12px;height:34px;align-items:center;font-size:12px;
}
.app-statusbar .sb-section{
    padding:0 12px;flex:1;display:flex;align-items:center;font-family:'JetBrains Mono',monospace;
    font-size:12px;color:#52525b;overflow:hidden;white-space:nowrap;
}
.app-statusbar .sb-section.sb-fixed{
    flex:0 0 auto;min-width:90px;text-align:center;justify-content:center;
    padding:3px 12px;background:rgba(255,20,147,.08);border-radius:6px;color:#FF69B4;font-weight:500;
}
.app-statusbar .sb-mode{
    flex:0 0 auto;min-width:120px;text-align:center;justify-content:center;
    padding:3px 12px;background:rgba(255,20,147,.08);border-radius:6px;color:#FF69B4;font-weight:500;
    font-family:'JetBrains Mono',monospace;font-size:12px;
}
#bbox-debug-count{font-family:'JetBrains Mono',monospace;font-size:12px;color:#52525b}

.dark .app-shell{background:#18181b}
.dark .upload-prompt-modern{background:transparent}
.dark .panel-card{background:#18181b}
.dark .settings-group{background:#18181b}
.dark .output-frame .out-title{color:#ffffff!important}
.dark .output-frame .out-title span{color:#ffffff!important}
.dark .out-download-btn{color:#FFB6C1!important}
.dark .out-download-btn:hover{color:#ffffff!important}

::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:#09090b}
::-webkit-scrollbar-thumb{background:#27272a;border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:#3f3f46}

@media(max-width:840px){
    .app-main-row{flex-direction:column}
    .app-main-right{width:100%}
    .app-main-left{border-right:none;border-bottom:1px solid #27272a}
    .mode-switcher{flex-wrap:wrap}
    .examples-grid{gap:8px;padding:10px 12px}
    .example-card{width:130px}
    .example-img-wrap img{height:80px}
}
"""

bbox_drawer_js = r"""
() => {
function init() {
    if (window.__bboxInitDone) return;

    const canvas     = document.getElementById('bbox-draw-canvas');
    const wrap       = document.getElementById('bbox-draw-wrap');
    const status     = document.getElementById('bbox-status');
    const badge      = document.getElementById('bbox-count');
    const debugCount = document.getElementById('bbox-debug-count');
    const jsonDisplay = document.getElementById('bbox-json-content');
    const moverHint  = document.getElementById('mover-box-hint');

    const btnDraw    = document.getElementById('tb-draw');
    const btnSelect  = document.getElementById('tb-select');
    const btnReset   = document.getElementById('tb-reset');
    const btnDel     = document.getElementById('tb-del');
    const btnUndo    = document.getElementById('tb-undo');
    const btnClear   = document.getElementById('tb-clear');
    const btnChange  = document.getElementById('tb-change-img');

    const uploadPrompt    = document.getElementById('upload-prompt');
    const uploadClickArea = document.getElementById('upload-click-area');
    const fileInput       = document.getElementById('custom-file-input');

    const modeRemover  = document.getElementById('mode-remover');
    const modeDesigner = document.getElementById('mode-designer');
    const modeMover    = document.getElementById('mode-mover');
    const runBtnEl     = document.getElementById('custom-run-btn');
    const runBtnLabel  = document.getElementById('run-btn-label');
    const statusMode   = document.getElementById('sb-mode-label');
    const hintBar      = document.getElementById('hint-bar-content');

    if (!canvas || !wrap || !debugCount || !btnDraw || !fileInput) {
        setTimeout(init, 250);
        return;
    }

    window.__bboxInitDone = true;
    const ctx = canvas.getContext('2d');

    let boxes = [];
    window.__bboxBoxes = boxes;
    window.__currentTaskMode = 'Object-Remover';

    let baseImg = null;
    let dispW = 512, dispH = 400;
    let selectedIdx = -1;
    let mode = 'draw';

    let dragging  = false;
    let dragType  = null;
    let dragStart = {x:0, y:0};
    let dragOrig  = null;
    const HANDLE  = 6;
    const RED_STROKE = 'rgba(255,0,0,0.95)';
    const RED_STROKE_WIDTH = 2;
    const SEL_STROKE = 'rgba(255,0,0,0.65)';
    let toastTimer = null;

    const DEFAULT_PROMPTS = {
        'Object-Remover': 'Remove the red highlighted object from the scene',
        'Design-Adder': 'Add the design pattern inside the red highlighted bounding box area',
        'Object-Mover': 'Move the object highlighted in the red box to the location indicated by the other red box in the scene'
    };
    const HINT_TEXTS = {
        'Object-Remover': '<b>Draw:</b> Click & drag to create selection boxes \u00b7 <b>Select:</b> Click a box to move or resize \u00b7 <kbd>Delete</kbd> removes selected \u00b7 <kbd>Clear</kbd> removes all \u00b7 <kbd>Reset</kbd> removes image',
        'Design-Adder': '<b>Draw:</b> Click & drag to create selection boxes \u00b7 <b>Select:</b> Click a box to move or resize \u00b7 <kbd>Delete</kbd> removes selected \u00b7 <kbd>Clear</kbd> removes all \u00b7 <kbd>Reset</kbd> removes image',
        'Object-Mover': '<span class="hint-mover-tag">MOVER</span> Draw <b>exactly 2</b> boxes: <b>Box 1 (SRC)</b> = object to move \u00b7 <b>Box 2 (DST)</b> = target location \u00b7 <kbd>Clear</kbd> to redraw \u00b7 Only 2 boxes allowed'
    };

    function showToast(message, type) {
        let toast = document.getElementById('app-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'app-toast';
            toast.className = 'toast-notification';
            toast.innerHTML = '<span class="toast-icon"></span><span class="toast-text"></span>';
            document.body.appendChild(toast);
        }
        const icon = toast.querySelector('.toast-icon');
        const text = toast.querySelector('.toast-text');
        toast.className = 'toast-notification ' + (type || 'error');
        if (type === 'warning') icon.textContent = '\u26A0';
        else if (type === 'info') icon.textContent = '\u2139';
        else icon.textContent = '\u2717';
        text.textContent = message;
        if (toastTimer) clearTimeout(toastTimer);
        void toast.offsetWidth;
        toast.classList.add('visible');
        toastTimer = setTimeout(function() { toast.classList.remove('visible'); }, 3500);
    }
    window.__showToast = showToast;

    function flashPromptError() {
        var pi = document.getElementById('custom-prompt-input');
        if (!pi) return;
        pi.classList.add('error-flash');
        pi.focus();
        setTimeout(function() { pi.classList.remove('error-flash'); }, 800);
    }

    function n2px(b) { return {x1:b.x1*dispW, y1:b.y1*dispH, x2:b.x2*dispW, y2:b.y2*dispH}; }
    function px2n(x1,y1,x2,y2) {
        return { x1:Math.min(x1,x2)/dispW, y1:Math.min(y1,y2)/dispH, x2:Math.max(x1,x2)/dispW, y2:Math.max(y1,y2)/dispH };
    }
    function clamp01(v){ return Math.max(0,Math.min(1,v)); }

    function fitSize(nw, nh) {
        var mw = wrap.clientWidth || 512, mh = 500;
        var r = Math.min(mw/nw, mh/nh, 1);
        dispW = Math.round(nw*r); dispH = Math.round(nh*r);
        canvas.width = dispW; canvas.height = dispH;
        canvas.style.width = dispW+'px'; canvas.style.height = dispH+'px';
    }
    function canvasXY(e) {
        var r = canvas.getBoundingClientRect();
        var cx = e.touches ? e.touches[0].clientX : e.clientX;
        var cy = e.touches ? e.touches[0].clientY : e.clientY;
        return {x:Math.max(0,Math.min(dispW,cx-r.left)), y:Math.max(0,Math.min(dispH,cy-r.top))};
    }

    function setGradioValue(containerId, value) {
        var container = document.getElementById(containerId);
        if (!container) return;
        container.querySelectorAll('input, textarea').forEach(function(el) {
            if (el.type === 'file' || el.type === 'range' || el.type === 'checkbox') return;
            var proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            var ns = Object.getOwnPropertyDescriptor(proto, 'value');
            if (ns && ns.set) {
                ns.set.call(el, value);
                el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            }
        });
    }
    window.__setGradioValue = setGradioValue;

    function formatJsonPretty(boxes) {
        if (!boxes || boxes.length === 0) return '[\n  // No bounding boxes defined\n]';
        var isMover = window.__currentTaskMode === 'Object-Mover';
        var lines = '[\n';
        boxes.forEach(function(b, i) {
            if (isMover) lines += '  // ' + (i === 0 ? 'SOURCE object' : 'TARGET location') + '\n';
            lines += '  {\n    "x1": '+b.x1.toFixed(4)+',\n    "y1": '+b.y1.toFixed(4)+',\n    "x2": '+b.x2.toFixed(4)+',\n    "y2": '+b.y2.toFixed(4)+'\n  }';
            if (i < boxes.length - 1) lines += ',';
            lines += '\n';
        });
        lines += ']';
        return lines;
    }

    function syncToGradio() {
        window.__bboxBoxes = boxes;
        var jsonStr = JSON.stringify(boxes);
        var isMover = window.__currentTaskMode === 'Object-Mover';
        if (debugCount) {
            if (isMover) {
                if (boxes.length === 0) debugCount.textContent = 'Mover: Draw source box';
                else if (boxes.length === 1) debugCount.textContent = 'Mover: Draw target box';
                else debugCount.textContent = 'Mover: SRC + DST ready';
            } else {
                debugCount.textContent = boxes.length > 0 ? boxes.length+' box'+(boxes.length>1?'es':'')+' drawn' : 'No boxes drawn';
            }
        }
        if (jsonDisplay) { jsonDisplay.textContent = formatJsonPretty(boxes); jsonDisplay.scrollTop = jsonDisplay.scrollHeight; }
        setGradioValue('boxes-json-input', jsonStr);
        updateMoverHint();
    }

    function updateMoverHint() {
        if (!moverHint) return;
        if (window.__currentTaskMode !== 'Object-Mover' || !baseImg) { moverHint.style.display='none'; return; }
        moverHint.style.display = 'block';
        if (boxes.length === 0) { moverHint.textContent = '\u25a0 Draw Box 1: Select the SOURCE object'; moverHint.style.color = '#FF69B4'; }
        else if (boxes.length === 1) { moverHint.textContent = '\u25a1 Draw Box 2: Mark the TARGET location'; moverHint.style.color = '#FFB6C1'; }
        else { moverHint.textContent = '\u2714 Both boxes ready \u2014 Source + Target defined'; moverHint.style.color = '#FF69B4'; }
    }

    function syncImageToGradio(dataUrl) { setGradioValue('hidden-image-b64', dataUrl); }
    function syncPromptToGradio() {
        var pi = document.getElementById('custom-prompt-input');
        if (pi) setGradioValue('prompt-gradio-input', pi.value);
    }
    function syncAdapterToGradio() { setGradioValue('adapter-choice-input', window.__currentTaskMode); }

    function clearExampleActiveStates() {
        document.querySelectorAll('.example-card').forEach(function(c) { c.classList.remove('example-active'); });
    }

    function showStatus(txt) { status.textContent = txt; status.style.display = 'block'; }
    function hideStatus() { status.style.display = 'none'; }
    window.__showStatus = showStatus;
    window.__hideStatus = hideStatus;

    function loadImageFromDataUrl(dataUrl) {
        var img = new window.Image();
        img.crossOrigin = 'anonymous';
        img.onload = function() {
            baseImg = img;
            boxes.length = 0;
            window.__bboxBoxes = boxes;
            selectedIdx = -1;
            fitSize(img.naturalWidth, img.naturalHeight);
            syncToGradio();
            redraw();
            hideStatus();
            if (uploadPrompt) uploadPrompt.style.display = 'none';
            syncImageToGradio(dataUrl);
        };
        img.onerror = function() {
            showToast('Failed to load image data', 'error');
        };
        img.src = dataUrl;
    }
    window.__loadImageFromDataUrl = loadImageFromDataUrl;

    function resetCanvas() {
        baseImg = null;
        boxes.length = 0;
        window.__bboxBoxes = boxes;
        selectedIdx = -1;
        dragging = false; dragType = null; dragOrig = null;
        fitSize(512, 400);
        syncToGradio(); syncImageToGradio('');
        redraw(); hideStatus();
        if (uploadPrompt) uploadPrompt.style.display = '';
        clearExampleActiveStates();
        showStatus('Image removed');
        setTimeout(hideStatus, 1500);
    }

    function switchTaskMode(taskMode) {
        window.__currentTaskMode = taskMode;
        var pi = document.getElementById('custom-prompt-input');
        [modeRemover, modeDesigner, modeMover].forEach(function(btn) { if(btn) btn.classList.remove('active'); });
        if (taskMode === 'Object-Remover') {
            if (modeRemover) modeRemover.classList.add('active');
            if (pi) pi.value = DEFAULT_PROMPTS['Object-Remover'];
            if (runBtnEl) { runBtnEl.classList.remove('design-mode','mover-mode'); }
            if (runBtnLabel) runBtnLabel.textContent = 'Remove Object';
            if (statusMode) statusMode.textContent = 'Object Remover';
        } else if (taskMode === 'Design-Adder') {
            if (modeDesigner) modeDesigner.classList.add('active');
            if (pi) pi.value = DEFAULT_PROMPTS['Design-Adder'];
            if (runBtnEl) { runBtnEl.classList.remove('mover-mode'); runBtnEl.classList.add('design-mode'); }
            if (runBtnLabel) runBtnLabel.textContent = 'Add Design';
            if (statusMode) statusMode.textContent = 'Design Adder';
        } else if (taskMode === 'Object-Mover') {
            if (modeMover) modeMover.classList.add('active');
            if (pi) pi.value = DEFAULT_PROMPTS['Object-Mover'];
            if (runBtnEl) { runBtnEl.classList.remove('design-mode'); runBtnEl.classList.add('mover-mode'); }
            if (runBtnLabel) runBtnLabel.textContent = 'Move Object';
            if (statusMode) statusMode.textContent = 'Object Mover';
        }
        if (hintBar) hintBar.innerHTML = HINT_TEXTS[taskMode] || HINT_TEXTS['Object-Remover'];
        syncPromptToGradio(); syncAdapterToGradio(); updateMoverHint(); redraw();
    }

    if (modeRemover) modeRemover.addEventListener('click', function() { switchTaskMode('Object-Remover'); });
    if (modeDesigner) modeDesigner.addEventListener('click', function() { switchTaskMode('Design-Adder'); });
    if (modeMover) modeMover.addEventListener('click', function() { switchTaskMode('Object-Mover'); });

    function getBoxLabel(index) {
        if (window.__currentTaskMode === 'Object-Mover') return index === 0 ? 'SRC' : 'DST';
        return '#'+(index+1);
    }

    function redraw(tempRect) {
        ctx.clearRect(0,0,dispW,dispH);
        if (!baseImg) { ctx.fillStyle='#09090b'; ctx.fillRect(0,0,dispW,dispH); updateBadge(); return; }
        ctx.drawImage(baseImg, 0, 0, dispW, dispH);
        boxes.forEach(function(b,i) {
            var p = n2px(b);
            var lx=p.x1,ty=p.y1,w=p.x2-p.x1,h=p.y2-p.y1;
            if (i === selectedIdx) { ctx.strokeStyle=SEL_STROKE; ctx.lineWidth=RED_STROKE_WIDTH+1; ctx.setLineDash([4,3]); }
            else { ctx.strokeStyle=RED_STROKE; ctx.lineWidth=RED_STROKE_WIDTH; ctx.setLineDash([]); }
            ctx.strokeRect(lx,ty,w,h); ctx.setLineDash([]);
            var label = getBoxLabel(i);
            ctx.fillStyle = '#FF0000'; ctx.font = 'bold 11px Inter,system-ui,sans-serif';
            ctx.textAlign = 'left'; ctx.textBaseline = 'top';
            var tw = ctx.measureText(label).width;
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(lx,ty-18,tw+10,18,3); else ctx.rect(lx,ty-18,tw+10,18);
            ctx.fill();
            ctx.fillStyle = '#fff'; ctx.fillText(label,lx+5,ty-15);
            if (i === selectedIdx) drawHandles(p);
        });
        if (tempRect) {
            var rx=Math.min(tempRect.x1,tempRect.x2),ry=Math.min(tempRect.y1,tempRect.y2);
            var rw=Math.abs(tempRect.x2-tempRect.x1),rh=Math.abs(tempRect.y2-tempRect.y1);
            ctx.strokeStyle=RED_STROKE; ctx.lineWidth=RED_STROKE_WIDTH; ctx.setLineDash([4,3]);
            ctx.strokeRect(rx,ry,rw,rh); ctx.setLineDash([]);
        }
        updateBadge();
    }

    function drawHandles(p) {
        var pts = handlePoints(p);
        for (var k in pts) {
            ctx.fillStyle='#FF0000'; ctx.beginPath(); ctx.arc(pts[k].x,pts[k].y,HANDLE,0,Math.PI*2); ctx.fill();
            ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.beginPath(); ctx.arc(pts[k].x,pts[k].y,HANDLE,0,Math.PI*2); ctx.stroke();
        }
    }
    function handlePoints(p) {
        var mx=(p.x1+p.x2)/2,my=(p.y1+p.y2)/2;
        return {tl:{x:p.x1,y:p.y1},tc:{x:mx,y:p.y1},tr:{x:p.x2,y:p.y1},ml:{x:p.x1,y:my},mr:{x:p.x2,y:my},bl:{x:p.x1,y:p.y2},bc:{x:mx,y:p.y2},br:{x:p.x2,y:p.y2}};
    }
    function hitHandle(px,py,boxIdx) {
        if (boxIdx<0) return null;
        var pts=handlePoints(n2px(boxes[boxIdx]));
        for (var k in pts) { if (Math.abs(px-pts[k].x)<=HANDLE+2 && Math.abs(py-pts[k].y)<=HANDLE+2) return k; }
        return null;
    }
    function hitBox(px,py) {
        for (var i=boxes.length-1;i>=0;i--) { var p=n2px(boxes[i]); if(px>=p.x1&&px<=p.x2&&py>=p.y1&&py<=p.y2) return i; }
        return -1;
    }
    function updateBadge() {
        if (boxes.length>0) {
            badge.style.display='block';
            badge.textContent = (window.__currentTaskMode==='Object-Mover' ? boxes.length+'/2 ' : boxes.length+' ')+'box'+(boxes.length>1?'es':'');
            badge.style.color='#FF1493'; badge.style.borderColor='rgba(255,20,147,.3)';
        } else badge.style.display='none';
    }
    function setMode(m) {
        mode = m;
        btnDraw.classList.toggle('active', m==='draw');
        btnSelect.classList.toggle('active', m==='select');
        canvas.style.cursor = m==='draw' ? 'crosshair' : 'default';
        if (m==='draw') selectedIdx = -1;
        redraw();
    }

    function onDown(e) {
        if (!baseImg) return;
        e.preventDefault();
        var pt = canvasXY(e);
        if (mode === 'draw') {
            if (window.__currentTaskMode==='Object-Mover' && boxes.length>=2) { showToast('Object Mover allows exactly 2 boxes. Clear to redraw.','warning'); return; }
            dragging=true; dragType='new'; dragStart=pt; selectedIdx=-1;
        } else {
            if (selectedIdx>=0) { var h=hitHandle(pt.x,pt.y,selectedIdx); if(h){dragging=true;dragType=h;dragStart=pt;dragOrig={...boxes[selectedIdx]};showStatus('Resizing '+getBoxLabel(selectedIdx));return;} }
            var hi=hitBox(pt.x,pt.y);
            if (hi>=0) {
                selectedIdx=hi;
                var h2=hitHandle(pt.x,pt.y,selectedIdx);
                if(h2){dragging=true;dragType=h2;dragStart=pt;dragOrig={...boxes[selectedIdx]};showStatus('Resizing '+getBoxLabel(selectedIdx));redraw();return;}
                dragging=true;dragType='move';dragStart=pt;dragOrig={...boxes[selectedIdx]};showStatus('Moving '+getBoxLabel(selectedIdx));
            } else { selectedIdx=-1; hideStatus(); }
            redraw();
        }
    }
    function onMove(e) {
        if (!baseImg) return;
        e.preventDefault();
        var pt = canvasXY(e);
        if (!dragging) {
            if (mode==='select') {
                if (selectedIdx>=0 && hitHandle(pt.x,pt.y,selectedIdx)) {
                    var h=hitHandle(pt.x,pt.y,selectedIdx);
                    var c={tl:'nwse-resize',tr:'nesw-resize',bl:'nesw-resize',br:'nwse-resize',tc:'ns-resize',bc:'ns-resize',ml:'ew-resize',mr:'ew-resize'};
                    canvas.style.cursor=c[h]||'move';
                } else canvas.style.cursor=hitBox(pt.x,pt.y)>=0?'move':'default';
            }
            return;
        }
        if (dragType==='new') { redraw({x1:dragStart.x,y1:dragStart.y,x2:pt.x,y2:pt.y}); showStatus(Math.abs(pt.x-dragStart.x).toFixed(0)+'\u00d7'+Math.abs(pt.y-dragStart.y).toFixed(0)+' px'); return; }
        var dx=(pt.x-dragStart.x)/dispW, dy=(pt.y-dragStart.y)/dispH;
        var b=boxes[selectedIdx], o=dragOrig;
        if (dragType==='move') {
            var bw=o.x2-o.x1,bh=o.y2-o.y1; var nx1=clamp01(o.x1+dx),ny1=clamp01(o.y1+dy);
            if(nx1+bw>1) nx1=1-bw; if(ny1+bh>1) ny1=1-bh;
            b.x1=nx1;b.y1=ny1;b.x2=nx1+bw;b.y2=ny1+bh;
        } else {
            var t=dragType;
            if(t.includes('l')) b.x1=clamp01(o.x1+dx); if(t.includes('r')) b.x2=clamp01(o.x2+dx);
            if(t.includes('t')) b.y1=clamp01(o.y1+dy); if(t.includes('b')) b.y2=clamp01(o.y2+dy);
            if(Math.abs(b.x2-b.x1)<0.01){b.x1=o.x1;b.x2=o.x2;} if(Math.abs(b.y2-b.y1)<0.01){b.y1=o.y1;b.y2=o.y2;}
            if(b.x1>b.x2){var t2=b.x1;b.x1=b.x2;b.x2=t2;} if(b.y1>b.y2){var t2=b.y1;b.y1=b.y2;b.y2=t2;}
        }
        redraw();
    }
    function onUp(e) {
        if (!dragging) return;
        if (e) e.preventDefault();
        dragging = false;
        if (dragType==='new') {
            var pt = e ? canvasXY(e) : dragStart;
            if (Math.abs(pt.x-dragStart.x)>4 && Math.abs(pt.y-dragStart.y)>4) {
                boxes.push(px2n(dragStart.x,dragStart.y,pt.x,pt.y));
                window.__bboxBoxes = boxes;
                selectedIdx = boxes.length-1;
                if (window.__currentTaskMode==='Object-Mover') showStatus((boxes.length===1?'Source object (SRC)':'Target location (DST)')+' marked');
                else showStatus('Box #'+boxes.length+' created');
            } else hideStatus();
        } else showStatus(getBoxLabel(selectedIdx)+' updated');
        dragType=null; dragOrig=null; syncToGradio(); redraw();
    }

    canvas.addEventListener('mousedown', onDown);
    canvas.addEventListener('mousemove', onMove);
    canvas.addEventListener('mouseup', onUp);
    canvas.addEventListener('mouseleave', function(e){if(dragging)onUp(e);});
    canvas.addEventListener('touchstart', onDown, {passive:false});
    canvas.addEventListener('touchmove', onMove, {passive:false});
    canvas.addEventListener('touchend', onUp, {passive:false});
    canvas.addEventListener('touchcancel', function(e){e.preventDefault();dragging=false;redraw();},{passive:false});

    function processFiles(files) {
        Array.from(files).forEach(function(file) {
            if (!file.type.startsWith('image/')) return;
            var reader = new FileReader();
            reader.onload = function(ev) {
                clearExampleActiveStates();
                loadImageFromDataUrl(ev.target.result);
                showStatus('Image loaded');
                setTimeout(hideStatus, 1500);
            };
            reader.readAsDataURL(file);
        });
    }

    fileInput.addEventListener('change', function(e) { processFiles(e.target.files); e.target.value = ''; });
    if (uploadClickArea) uploadClickArea.addEventListener('click', function() { fileInput.click(); });
    if (btnChange) btnChange.addEventListener('click', function() { fileInput.click(); });

    wrap.addEventListener('dragover', function(e) { e.preventDefault(); wrap.classList.add('drag-over'); });
    wrap.addEventListener('dragleave', function(e) { e.preventDefault(); wrap.classList.remove('drag-over'); });
    wrap.addEventListener('drop', function(e) {
        e.preventDefault(); wrap.classList.remove('drag-over');
        if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files);
    });

    btnDraw.addEventListener('click', function(){setMode('draw');});
    btnSelect.addEventListener('click', function(){setMode('select');});
    btnReset.addEventListener('click', resetCanvas);
    btnDel.addEventListener('click', function() {
        if (selectedIdx>=0 && selectedIdx<boxes.length) {
            var removed=getBoxLabel(selectedIdx); boxes.splice(selectedIdx,1); window.__bboxBoxes=boxes;
            selectedIdx=-1; syncToGradio(); redraw(); showStatus(removed+' deleted');
        } else showStatus('Select a box first');
    });
    btnUndo.addEventListener('click', function() {
        if (boxes.length>0) { boxes.pop(); window.__bboxBoxes=boxes; selectedIdx=-1; syncToGradio(); redraw(); showStatus('Last box removed'); }
    });
    btnClear.addEventListener('click', function() {
        boxes.length=0; window.__bboxBoxes=boxes; selectedIdx=-1; syncToGradio(); redraw(); hideStatus();
    });

    var promptInput = document.getElementById('custom-prompt-input');
    if (promptInput) promptInput.addEventListener('input', function() { promptInput.classList.remove('error-flash'); syncPromptToGradio(); });

    function syncSlider(customId, gradioId) {
        var slider = document.getElementById(customId);
        var valSpan = document.getElementById(customId+'-val');
        if (!slider) return;
        slider.addEventListener('input', function() {
            if (valSpan) valSpan.textContent = slider.value;
            var container = document.getElementById(gradioId);
            if (!container) return;
            container.querySelectorAll('input[type="range"],input[type="number"]').forEach(function(el) {
                var ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                if (ns && ns.set) { ns.set.call(el, slider.value); el.dispatchEvent(new Event('input',{bubbles:true,composed:true})); el.dispatchEvent(new Event('change',{bubbles:true,composed:true})); }
            });
        });
    }
    syncSlider('custom-seed','gradio-seed');
    syncSlider('custom-guidance','gradio-guidance');
    syncSlider('custom-steps','gradio-steps');
    syncSlider('custom-height','gradio-height');
    syncSlider('custom-width','gradio-width');

    var randCheck = document.getElementById('custom-randomize');
    if (randCheck) {
        randCheck.addEventListener('change', function() {
            var container = document.getElementById('gradio-randomize');
            if (!container) return;
            var cb = container.querySelector('input[type="checkbox"]');
            if (cb && cb.checked !== randCheck.checked) cb.click();
        });
    }

    function showLoaders() {
        var l1=document.getElementById('output-loader'),l2=document.getElementById('preview-loader');
        if(l1)l1.classList.add('active'); if(l2)l2.classList.add('active');
        var sb=document.querySelector('.sb-fixed'); if(sb)sb.textContent='Processing...';
        if(window.__loaderTimeout) clearTimeout(window.__loaderTimeout);
        window.__loaderTimeout=setTimeout(hideLoaders,120000);
    }
    function hideLoaders() {
        var l1=document.getElementById('output-loader'),l2=document.getElementById('preview-loader');
        if(l1)l1.classList.remove('active'); if(l2)l2.classList.remove('active');
        var sb=document.querySelector('.sb-fixed'); if(sb)sb.textContent='Done';
        if(window.__loaderTimeout){clearTimeout(window.__loaderTimeout);window.__loaderTimeout=null;}
    }
    window.__showLoaders = showLoaders;
    window.__hideLoaders = hideLoaders;

    window.__clickGradioRunBtn = function() {
        var pi = document.getElementById('custom-prompt-input');
        var promptVal = pi ? pi.value.trim() : '';
        if (!baseImg) { showToast('Please upload an image first','error'); return; }
        if (!promptVal) { showToast('Please enter an edit prompt','warning'); flashPromptError(); return; }
        if (boxes.length===0) { showToast('Please draw at least one bounding box','warning'); return; }
        if (window.__currentTaskMode==='Object-Mover') {
            if (boxes.length<2) { showToast('Object Mover needs 2 boxes. Draw a target box.','warning'); return; }
            if (boxes.length>2) { showToast('Object Mover needs exactly 2 boxes. Clear and redraw.','warning'); return; }
        }
        syncPromptToGradio(); syncAdapterToGradio(); syncToGradio(); showLoaders();
        setTimeout(function() {
            var gradioBtn=document.getElementById('gradio-run-btn');
            if(!gradioBtn) return;
            var btn=gradioBtn.querySelector('button');
            if(btn) btn.click(); else gradioBtn.click();
        }, 200);
    };

    if (runBtnEl) runBtnEl.addEventListener('click', function() { window.__clickGradioRunBtn(); });

    /* ── Example Cards: Gradio Callback Pattern ── */
    document.querySelectorAll('.example-card[data-example-idx]').forEach(function(card) {
        card.addEventListener('click', function() {
            var idx = card.getAttribute('data-example-idx');

            document.querySelectorAll('.example-card.example-loading').forEach(function(c) { c.classList.remove('example-loading'); });
            clearExampleActiveStates();
            card.classList.add('example-loading');

            showToast('Loading example...', 'info');

            setGradioValue('example-result-data', '');
            setGradioValue('example-idx-input', idx);

            setTimeout(function() {
                var btn = document.getElementById('example-load-btn');
                if (btn) {
                    var b = btn.querySelector('button');
                    if (b) b.click(); else btn.click();
                }
            }, 150);

            setTimeout(function() { card.classList.remove('example-loading'); }, 12000);
        });
    });

    new ResizeObserver(function() {
        if (baseImg) { fitSize(baseImg.naturalWidth, baseImg.naturalHeight); redraw(); }
    }).observe(wrap);

    setMode('draw'); fitSize(512,400); redraw(); syncToGradio(); syncAdapterToGradio(); updateMoverHint();
}
init();
}
"""

wire_outputs_js = r"""
() => {
function watchOutputs() {
    var resultContainer = document.getElementById('gradio-result');
    var previewContainer = document.getElementById('gradio-preview');
    var outBody = document.getElementById('output-image-container');
    var prevBody = document.getElementById('preview-image-container');
    var outPh = document.getElementById('output-placeholder');
    var prevPh = document.getElementById('preview-placeholder');
    var dlBtnOut = document.getElementById('dl-btn-output');
    var dlBtnPrev = document.getElementById('dl-btn-preview');

    if (!resultContainer || !previewContainer || !outBody || !prevBody) { setTimeout(watchOutputs, 500); return; }

    function downloadImage(src, name) {
        var a=document.createElement('a'); a.href=src; a.download=name||'image.png';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
    }
    if (dlBtnOut) dlBtnOut.addEventListener('click', function(e) { e.stopPropagation(); var img=outBody.querySelector('img.modern-out-img'); if(img&&img.src) downloadImage(img.src,'output_result.png'); });
    if (dlBtnPrev) dlBtnPrev.addEventListener('click', function(e) { e.stopPropagation(); var img=prevBody.querySelector('img.modern-out-img'); if(img&&img.src) downloadImage(img.src,'input_preview.png'); });

    function syncImages() {
        var ri=resultContainer.querySelector('img');
        if (ri && ri.src) {
            if(outPh) outPh.style.display='none';
            var ex=outBody.querySelector('img.modern-out-img');
            if(!ex){ex=document.createElement('img');ex.className='modern-out-img';outBody.appendChild(ex);}
            if(ex.src!==ri.src){ex.src=ri.src; if(dlBtnOut)dlBtnOut.classList.add('visible'); if(window.__hideLoaders)window.__hideLoaders();}
        }
        var pi=previewContainer.querySelector('img');
        if (pi && pi.src) {
            if(prevPh) prevPh.style.display='none';
            var ex2=prevBody.querySelector('img.modern-out-img');
            if(!ex2){ex2=document.createElement('img');ex2.className='modern-out-img';prevBody.appendChild(ex2);}
            if(ex2.src!==pi.src){ex2.src=pi.src; if(dlBtnPrev)dlBtnPrev.classList.add('visible');}
        }
    }
    var obs = new MutationObserver(syncImages);
    obs.observe(resultContainer, {childList:true,subtree:true,attributes:true,attributeFilter:['src']});
    obs.observe(previewContainer, {childList:true,subtree:true,attributes:true,attributeFilter:['src']});
    setInterval(syncImages, 800);
}
watchOutputs();

function watchDimensions() {
    var wC=document.getElementById('gradio-width'),hC=document.getElementById('gradio-height');
    var wS=document.getElementById('custom-width'),hS=document.getElementById('custom-height');
    var wV=document.getElementById('custom-width-val'),hV=document.getElementById('custom-height-val');
    if(!wC||!hC||!wS||!hS){setTimeout(watchDimensions,500);return;}
    function sync(){
        var wi=wC.querySelector('input[type="range"],input[type="number"]');
        var hi=hC.querySelector('input[type="range"],input[type="number"]');
        if(wi&&wi.value){wS.value=wi.value;if(wV)wV.textContent=wi.value;}
        if(hi&&hi.value){hS.value=hi.value;if(hV)hV.textContent=hi.value;}
    }
    var o=new MutationObserver(sync);
    o.observe(wC,{childList:true,subtree:true,attributes:true,attributeFilter:['value']});
    o.observe(hC,{childList:true,subtree:true,attributes:true,attributeFilter:['value']});
    setInterval(sync,1000);
}
watchDimensions();

function watchSeed() {
    var sC=document.getElementById('gradio-seed'),sS=document.getElementById('custom-seed'),sV=document.getElementById('custom-seed-val');
    if(!sC||!sS){setTimeout(watchSeed,500);return;}
    function sync(){var el=sC.querySelector('input[type="range"],input[type="number"]');if(el&&el.value){sS.value=el.value;if(sV)sV.textContent=el.value;}}
    var o=new MutationObserver(sync);
    o.observe(sC,{childList:true,subtree:true,attributes:true,attributeFilter:['value']});
    setInterval(sync,1000);
}
watchSeed();

function watchExampleResults() {
    var container = document.getElementById('example-result-data');
    if (!container) { setTimeout(watchExampleResults, 500); return; }
    var lastProcessed = '';
    function checkResult() {
        var el = container.querySelector('textarea') || container.querySelector('input');
        if (!el) return;
        var val = el.value;
        if (!val || val === lastProcessed || val.length < 20) return;
        try {
            var data = JSON.parse(val);
            if (data.status === 'ok' && data.image) {
                lastProcessed = val;
                document.querySelectorAll('.example-card.example-active').forEach(function(c) { c.classList.remove('example-active'); });
                if (window.__loadImageFromDataUrl) window.__loadImageFromDataUrl(data.image);
                document.querySelectorAll('.example-card.example-loading').forEach(function(c) {
                    c.classList.remove('example-loading');
                    c.classList.add('example-active');
                });
                if (window.__showToast) window.__showToast('Example loaded', 'info');
            } else if (data.status === 'error') {
                document.querySelectorAll('.example-card.example-loading').forEach(function(c) { c.classList.remove('example-loading'); });
                if (window.__showToast) window.__showToast('Could not load example image', 'error');
            }
        } catch(e) { console.error('Example parse error:', e); }
    }
    var obs = new MutationObserver(checkResult);
    obs.observe(container, {childList:true, subtree:true, characterData:true, attributes:true});
    setInterval(checkResult, 500);
}
watchExampleResults();
}
"""

DOWNLOAD_SVG = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 16l-5-5h3V4h4v7h3l-5 5z"/><path d="M20 18H4v2h16v-2z"/></svg>'

with gr.Blocks(css=css) as demo:

    hidden_image_b64 = gr.Textbox(elem_id="hidden-image-b64", elem_classes="hidden-input", container=False)
    boxes_json = gr.Textbox(value="[]", elem_id="boxes-json-input", elem_classes="hidden-input", container=False)
    prompt = gr.Textbox(value=DEFAULT_PROMPTS["Object-Remover"], elem_id="prompt-gradio-input", elem_classes="hidden-input", container=False)
    adapter_choice = gr.Textbox(value="Object-Remover", elem_id="adapter-choice-input", elem_classes="hidden-input", container=False)
    seed = gr.Slider(minimum=0, maximum=MAX_SEED, step=1, value=0, elem_id="gradio-seed", elem_classes="hidden-input", container=False)
    randomize_seed = gr.Checkbox(value=True, elem_id="gradio-randomize", elem_classes="hidden-input", container=False)
    guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=1.0, elem_id="gradio-guidance", elem_classes="hidden-input", container=False)
    num_inference_steps = gr.Slider(minimum=1, maximum=20, step=1, value=4, elem_id="gradio-steps", elem_classes="hidden-input", container=False)
    height_slider = gr.Slider(minimum=256, maximum=2048, step=8, value=1024, elem_id="gradio-height", elem_classes="hidden-input", container=False)
    width_slider = gr.Slider(minimum=256, maximum=2048, step=8, value=1024, elem_id="gradio-width", elem_classes="hidden-input", container=False)
    result = gr.Image(elem_id="gradio-result", elem_classes="hidden-input", container=False, format="png")
    preview = gr.Image(elem_id="gradio-preview", elem_classes="hidden-input", container=False)

    example_idx = gr.Textbox(value="", elem_id="example-idx-input", elem_classes="hidden-input", container=False)
    example_result = gr.Textbox(value="", elem_id="example-result-data", elem_classes="hidden-input", container=False)
    example_load_btn = gr.Button("Load Example", elem_id="example-load-btn")

    gr.HTML(f"""
    <div class="app-shell">

        <div class="app-header">
            <div class="app-header-left">
                <div class="app-logo">\u2b1a</div>
                <span class="app-title">QIE-Bbox-Studio</span>
                <span class="app-badge">Bbox</span>
            </div>
            <div class="mode-switcher">
                <button id="mode-remover" class="mode-btn active" title="Object Removal Mode">
                    <span class="mode-icon">\u2326</span> Object Remover
                </button>
                <button id="mode-designer" class="mode-btn" title="Design Adder Mode">
                    <span class="mode-icon">\u2726</span> Design Adder
                </button>
                <button id="mode-mover" class="mode-btn" title="Object Mover Mode">
                    <span class="mode-icon">\u21c4</span> Object Mover
                </button>
            </div>
        </div>

        <div class="app-toolbar">
            <button id="tb-draw" class="modern-tb-btn active" title="Draw bounding boxes">
                <span class="tb-icon">\u25ac</span><span class="tb-label">Draw</span>
            </button>
            <button id="tb-select" class="modern-tb-btn" title="Select, move, resize boxes">
                <span class="tb-icon">\u21c9</span><span class="tb-label">Select</span>
            </button>
            <button id="tb-reset" class="modern-tb-btn" title="Reset canvas and remove image">
                <span class="tb-icon">\u27f2</span><span class="tb-label">Reset</span>
            </button>
            <div class="tb-sep"></div>
            <button id="tb-del" class="modern-tb-btn" title="Delete selected box">
                <span class="tb-icon">\u2715</span><span class="tb-label">Delete</span>
            </button>
            <button id="tb-undo" class="modern-tb-btn" title="Undo last box">
                <span class="tb-icon">\u21a9</span><span class="tb-label">Undo</span>
            </button>
            <button id="tb-clear" class="modern-tb-btn" title="Clear all boxes">
                <span class="tb-icon">\u2716</span><span class="tb-label">Clear</span>
            </button>
            <div class="tb-sep"></div>
            <button id="tb-change-img" class="modern-tb-btn" title="Upload a different image">
                <span class="tb-label">Upload\u2026</span>
            </button>
        </div>

        <div class="app-main-row">
            <div class="app-main-left">
                <div id="bbox-draw-wrap">
                    <div id="upload-prompt" class="upload-prompt-modern">
                        <div id="upload-click-area" class="upload-click-area">
                            <svg viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <rect x="8" y="14" width="64" height="52" rx="6" fill="none" stroke="#FF1493" stroke-width="2" stroke-dasharray="4 3"/>
                                <polygon points="12,62 30,40 42,50 54,34 68,62" fill="rgba(255,20,147,0.15)" stroke="#FF1493" stroke-width="1.5"/>
                                <circle cx="28" cy="30" r="6" fill="rgba(255,20,147,0.2)" stroke="#FF1493" stroke-width="1.5"/>
                            </svg>
                            <span class="upload-main-text">Click or drag image here</span>
                            <span class="upload-sub-text">Supports JPG, PNG, WebP</span>
                        </div>
                    </div>
                    <input id="custom-file-input" type="file" accept="image/*" style="display:none;" />
                    <canvas id="bbox-draw-canvas" width="512" height="400"></canvas>
                    <div id="bbox-status"></div>
                    <div id="bbox-count"></div>
                    <div id="mover-box-hint"></div>
                </div>

                <div class="hint-bar" id="hint-bar-content">
                    <b>Draw:</b> Click &amp; drag to create selection boxes &nbsp;&middot;&nbsp;
                    <b>Select:</b> Click a box to move or resize &nbsp;&middot;&nbsp;
                    <kbd>Delete</kbd> removes selected &nbsp;&middot;&nbsp;
                    <kbd>Clear</kbd> removes all &nbsp;&middot;&nbsp;
                    <kbd>Reset</kbd> removes image
                </div>

                <div class="json-panel">
                    <div class="json-panel-title">Bounding Boxes</div>
                    <div class="json-panel-content" id="bbox-json-content">[
  // No bounding boxes defined
]</div>
                </div>

                {examples_html_block}
            </div>

            <div class="app-main-right">
                <div class="panel-card">
                    <div class="panel-card-title">Edit Instruction</div>
                    <div class="panel-card-body">
                        <label class="modern-label" for="custom-prompt-input">Prompt</label>
                        <textarea id="custom-prompt-input" class="modern-textarea" rows="2" placeholder="Describe the edit...">Remove the red highlighted object from the scene</textarea>
                    </div>
                </div>

                <div style="padding:12px 20px;">
                    <button id="custom-run-btn" class="btn-run">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M19 7l-7 5-7-5V5l7 5 7-5v2zm0 6l-7 5-7-5v-2l7 5 7-5v2z"/></svg>
                        <span id="run-btn-label">Remove Object</span>
                    </button>
                </div>

                <div class="output-frame" style="flex:1">
                    <div class="out-title">
                        <span>Output</span>
                        <span id="dl-btn-output" class="out-download-btn" title="Download">{DOWNLOAD_SVG} Save</span>
                    </div>
                    <div class="out-body" id="output-image-container">
                        <div class="modern-loader" id="output-loader">
                            <div class="loader-spinner"></div>
                            <div class="loader-text">Processing image\u2026</div>
                            <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
                        </div>
                        <div class="out-placeholder" id="output-placeholder">Result will appear here</div>
                    </div>
                </div>

                <div class="output-frame">
                    <div class="out-title">
                        <span>Input Preview</span>
                        <span id="dl-btn-preview" class="out-download-btn" title="Download">{DOWNLOAD_SVG} Save</span>
                    </div>
                    <div class="out-body" id="preview-image-container">
                        <div class="modern-loader" id="preview-loader">
                            <div class="loader-spinner"></div>
                            <div class="loader-text">Preparing input\u2026</div>
                            <div class="loader-bar-track"><div class="loader-bar-fill"></div></div>
                        </div>
                        <div class="out-placeholder" id="preview-placeholder">Preview will appear here</div>
                    </div>
                </div>

                <div class="settings-group">
                    <div class="settings-group-title">Advanced Settings</div>
                    <div class="settings-group-body">
                        <div class="slider-row">
                            <label>Seed</label>
                            <input type="range" id="custom-seed" min="0" max="2147483647" step="1" value="0">
                            <span class="slider-val" id="custom-seed-val">0</span>
                        </div>
                        <div class="checkbox-row">
                            <input type="checkbox" id="custom-randomize" checked>
                            <label for="custom-randomize">Randomize seed</label>
                        </div>
                        <div class="slider-row">
                            <label>Guidance</label>
                            <input type="range" id="custom-guidance" min="1" max="10" step="0.1" value="1.0">
                            <span class="slider-val" id="custom-guidance-val">1.0</span>
                        </div>
                        <div class="slider-row">
                            <label>Steps</label>
                            <input type="range" id="custom-steps" min="1" max="20" step="1" value="4">
                            <span class="slider-val" id="custom-steps-val">4</span>
                        </div>
                        <div class="slider-row">
                            <span class="dim-label">Width</span>
                            <input type="range" id="custom-width" min="256" max="2048" step="8" value="1024">
                            <span class="slider-val" id="custom-width-val">1024</span>
                        </div>
                        <div class="slider-row">
                            <span class="dim-label">Height</span>
                            <input type="range" id="custom-height" min="256" max="2048" step="8" value="1024">
                            <span class="slider-val" id="custom-height-val">1024</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="app-statusbar">
            <div class="sb-section" id="bbox-debug-count">No boxes drawn</div>
            <div class="sb-section sb-mode" id="sb-mode-label">Object Remover</div>
            <div class="sb-section sb-fixed">Ready</div>
        </div>
    </div>
    """)

    run_btn = gr.Button("Run", elem_id="gradio-run-btn")

    demo.load(fn=None, js=bbox_drawer_js)
    demo.load(fn=None, js=wire_outputs_js)

    run_btn.click(
        fn=infer_bbox_task,
        inputs=[hidden_image_b64, boxes_json, prompt, adapter_choice, seed, randomize_seed,
                guidance_scale, num_inference_steps, height_slider, width_slider],
        outputs=[result, seed, preview],
        js=r"""(b64, bj, p, ac, s, rs, gs, nis, h, w) => {
            var boxes = window.__bboxBoxes || [];
            var json = JSON.stringify(boxes);
            var taskMode = window.__currentTaskMode || 'Object-Remover';
            return [b64, json, p, taskMode, s, rs, gs, nis, h, w];
        }""",
    )

    hidden_image_b64.change(
        fn=update_dimensions_on_upload,
        inputs=[hidden_image_b64],
        outputs=[width_slider, height_slider],
    )

    example_load_btn.click(
        fn=load_example_data,
        inputs=[example_idx],
        outputs=[example_result],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue().launch(
        mcp_server=True,
        ssr_mode=False,
        show_error=True,
        allowed_paths=["examples"],
    )