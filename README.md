# mlx-bopbtl-swift

The **scratch detector** from Microsoft's *Bringing Old Photos Back to Life* (BOPBTL), ported to
MLX-Swift. Grayscale photo in → scratch/defect probability mask out — the missing *detection* half
of automatic damage repair (the registered `imageInpaint` capability fixes what this marks, no
manual brushing).

## Model

`UNet(in=1, out=1, depth=4, conv_num=2, wf=6, batch_norm, up_mode="upsample", antialiasing=True)` —
37,626,177 params, from the released `FT_Epoch_latest.pt` (`model_state`, Adam state discarded:
1.90 GiB zip → 150.6 MB fp32 safetensors). Reference behaviour: short-side-256 (/16-aligned)
grayscale input normalized to [-1,1]; `sigmoid(logits) ≥ 0.4` is upstream's mask threshold.

## Parity

Fixture gate (`bopbtl-gate`) vs fp32 PyTorch on three shapes incl. non-square:
**worst 119.8 dB, max |Δ| 4e-5** — effectively exact.

🚨 **The forward is pinned to the CPU stream, and the pin is a receipt, not a preference.** The
Metal fp32 `conv2d` kernel diverges (~10 bits lost, GPU-vs-CPU max |Δ| 0.08–0.12 on random data)
in a precise window: **inC ≥ 128 AND outC ≥ 128 AND spatial ≥ ~64** — which this UNet's interior
hits repeatedly. On CPU the whole 256-class forward is sub-second, so nothing is lost. Run
`bopbtl-gate --gpu-probe` after every mlx-swift bump; when the window reports clean, remove the pin
in `ScratchDetector.logits`.

## License

Port code MIT. Upstream code **and pretrained model** MIT (Microsoft) — the repo's LICENSE file,
every source header, and README §License all say MIT, with §License extending it to the pretrained
model verbatim; a README line saying "academic research use only" was reviewed against those formal
grants and cleared 2026-08-01. ⚠️ Microsoft has not released the training dataset — the model
cannot be retrained or audited; detection failures degrade to manual masking.
