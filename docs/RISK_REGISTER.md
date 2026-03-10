# Risk Register (MVP)

## Technical Risks
- Pretrained model underperforms on stable footage due to angle/lighting/occlusion
- DLC install friction (torch/CUDA/platform compatibility)
- Poor crop/scale reduces keypoint stability
- Stride proxy unreliable on short or noisy clips

## Operational Risks
- Jack footage too long/untrimmed for rapid iteration
- Camera motion introduces jitter mistaken for horse movement
- Multiple horses/handlers in frame confuse tracking

## Mitigations
- Start with known-good sample demo first
- Enforce capture protocol (side view, stable camera, single horse)
- Report confidence/jitter and label output quality Green/Yellow/Red
- Use manual crop or optional detector-based crop if needed
