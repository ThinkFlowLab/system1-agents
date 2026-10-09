#!/usr/bin/env bash
# Open-Jev-27B-v1.1 merged export for system1-omni's native worker (recipe/open_jev/native.md), on CPU.
set -euo pipefail
K=$HOME/kit; O=$K/system1-omni; export PATH=$HOME/.local/bin:$PATH
cd $O && git log -1 --format="system1-omni %H" | tee $K/export_versions.txt
[ -d $K/oj-env ] || uv venv -q -p 3.12 $K/oj-env
uv pip install -q -p $K/oj-env/bin/python "torch>=2.8" --index-url https://download.pytorch.org/whl/cpu
uv pip install -q -p $K/oj-env/bin/python transformers==5.10.2 peft==0.19.1 accelerate==1.13.0 safetensors "huggingface_hub[cli]"
$K/oj-env/bin/python -c "import torch,transformers,peft;print('torch',torch.__version__,'transformers',transformers.__version__,'peft',peft.__version__)" | tee -a $K/export_versions.txt
HF=$K/oj-env/bin/hf
$HF download Qwen/Qwen3.8-27B --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir weights/Qwen3.8-27B >/dev/null
$HF download ZefanCai/Open-Jev-27B-v1.1 --revision 28cf73067d5b337860bbef3c85b8b82ba8730956 \
  --include 'package/checkpoint/*' --local-dir weights/Open-Jev-27B-v1.1 >/dev/null
du -sh weights/* | tee -a $K/export_versions.txt
echo "== export start $(date -u +%H:%M:%S)"
/usr/bin/time -v env CUDA_VISIBLE_DEVICES='' $K/oj-env/bin/python recipe/open_jev/export_merged.py \
  --base weights/Qwen3.8-27B --checkpoint weights/Open-Jev-27B-v1.1/package/checkpoint --out weights/open-jev-27b-merged \
  2>&1 | tail -30
du -sh weights/open-jev-27b-merged | tee -a $K/export_versions.txt
echo "== EXPORT_DONE $(date -u +%H:%M:%S)"
