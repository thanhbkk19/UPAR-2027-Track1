UPAR Track 1 v18 mixed research candidate

PE 336x168 / EVA 196x196, each blending 25% original head and 75% robust head in probability space.
Same original v11 WiSE backbone training on 117,970 organizer training images.
Robust readouts use per-source mixup and a VLM-trained readout on 16 attributes.
VLM targets came from 24,000 organizer training images. No VLM runs at inference.
The global positive calibration bias is removed; microbatch size is 32.
No hidden data, source routing, network access, or batch-dependent adaptation.

PE-Core-L/14, Meta Perception Encoder: Apache-2.0.
https://huggingface.co/timm/vit_pe_core_large_patch14_336.fb
EVA02-L/14 CLIP, BAAI: MIT.
https://huggingface.co/timm/eva02_large_patch14_clip_224.merged2b
Training-only teacher: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
Organizer annotations: https://github.com/speckean/UPAR-Challenge-2027
Underlying datasets retain their original terms.
