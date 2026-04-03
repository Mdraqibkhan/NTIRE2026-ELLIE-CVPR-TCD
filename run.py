# NTIRE 2026 - Greya ScaleQuery (Reduced) Inference Script
# Author: Raqib Hassan Khan 
# To run: python run.py --input_dir ./low --model_path ./model.pth --output_dir submission_ntire --zip ntire2026_submission.zip
import argparse, os, zipfile
import torch
from PIL import Image
import torchvision.transforms as transforms
from greya_scalequery_reduced import my_model_lite

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',  default='./low', help='folder of input (dark) images')
parser.add_argument('--model_path', default='./model.pth')
parser.add_argument('--output_dir', default='submission_ntire')
parser.add_argument('--zip',        default='ntire2026_submission.zip')
parser.add_argument('--no_cuda',    action='store_true')
opt = parser.parse_args()

device = torch.device('cuda:0' if not opt.no_cuda and torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')

# ── Load model ────────────────────────────────────────────────────────────────
model = my_model_lite()
state = torch.load(opt.model_path, map_location='cpu')
state = {k: v.float() if v.dtype == torch.float16 else v for k, v in state.items()}
model.load_state_dict(state)
model = model.to(device).eval()

total   = sum(p.numel() for p in model.parameters())
size_mb = os.path.getsize(opt.model_path) / 1e6
print(f'Parameters : {total/10**6:,.3f}M')
print(f'Model file : {size_mb:.3f} MB  {"✅ ≤ 1 MB" if size_mb <= 1.0 else "❌ EXCEEDS 1 MB!"}')

# ── Helpers ───────────────────────────────────────────────────────────────────
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def tensor_to_pil(t):
    if t.dim() == 4: t = t.squeeze(0)
    arr = ((t.cpu().clamp(-1, 1) + 1.0) * 127.5).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy())

def enhance(pil_img):
    W, H = pil_img.size
    pH = ((H + 7) // 8) * 8
    pW = ((W + 7) // 8) * 8
    padded = pil_img.resize((pW, pH), Image.BILINEAR) if (pH != H or pW != W) else pil_img
    inp = to_tensor(padded).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp, inp)
    return tensor_to_pil(out).resize((W, H), Image.BILINEAR)

# ── Inference loop ────────────────────────────────────────────────────────────
valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
img_files = sorted([f for f in os.listdir(opt.input_dir) if f.lower().endswith(valid_ext)])
assert img_files, f"No images in {opt.input_dir}"

os.makedirs(opt.output_dir, exist_ok=True)
print(f'\nProcessing {len(img_files)} images...\n')

for i, fname in enumerate(img_files, 1):
    img     = Image.open(os.path.join(opt.input_dir, fname)).convert('RGB')
    W, H    = img.size
    out_img = enhance(img)
    out_img.save(os.path.join(opt.output_dir, fname))
    print(f'  [{i:4d}/{len(img_files)}] {fname}  ({W}×{H})')

# ── Pack ZIP ──────────────────────────────────────────────────────────────────
with zipfile.ZipFile(opt.zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in img_files:
        zf.write(os.path.join(opt.output_dir, fname), arcname=fname)

print(f'\n✅ Submission ZIP : {opt.zip}  ({os.path.getsize(opt.zip)/1e6:.1f} MB)')
print(f'   {len(img_files)} images | CodaBench ready')





