import sys, os, torch, json
sys.path.insert(0, '.')
sys.path.append('./MobileVLM_code')
from config import get_config
from PIL import Image
from utils.evaluation import _generate

cfg = get_config()

from transformers import AutoTokenizer, CLIPImageProcessor
from mobilevlm.model.mobilellama import MobileLlamaForCausalLM

snapshot = './model_cache/models--mtgv--MobileVLM_V2-1.7B/snapshots/9a5b623a83feae6a6b2ecad7a843334ccc119ce1'
tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=cfg.paths.model_cache_dir)
model = MobileLlamaForCausalLM.from_pretrained(snapshot, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0}, ignore_mismatched_sizes=True)
model.eval()
model.get_model().vision_tower.load_model()
model.get_model().vision_tower.vision_tower.to(dtype=torch.bfloat16, device="cuda")

with open('data/pope/coco_pope_adversarial.json') as f:
    data = [json.loads(line) for line in f if line.strip()]

for item in data[:5]:
    img_path = os.path.join('data/coco/val2014', item['image'])
    if not os.path.exists(img_path):
        print(f"MISSING: {img_path}")
        continue
    image = Image.open(img_path).convert("RGB")
    answer = _generate(model, tokenizer, image_processor, image, item['text'], 'cuda')
    pred = "yes" if "yes" in answer else "no"
    print(f"Q: {item['text']}")
    print(f"  RAW answer: repr='{repr(answer)}'")
    print(f"  Parsed: {pred} | GT: {item['label']}")
    print()
