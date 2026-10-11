# sd-webui-anima-dc

An extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
that runs **Anima-DC** models: [Anima](https://huggingface.co/circlestone-labs/Anima) moved onto a
deeply compressed latent space with the [DC-Gen](https://arxiv.org/abs/2509.25180) recipe, such as
[Battera-Anima-DC](https://huggingface.co/FA770/Battera-Anima-DC).

These models use a 32× compressed autoencoder instead of the Qwen-Image VAE, so Forge Neo cannot
load them on its own.

## Installation

In Forge Neo, open **Extensions → Install from URL**, enter

```
https://github.com/FA7770/sd-webui-anima-dc
```

and click **Install**. Then restart the WebUI (a page reload is not enough).

The console shows this line at startup when the extension is active:

```
[Anima-DC] loader hooks installed
```

If you install manually, the folder must be `extensions/sd-webui-anima-dc/scripts/...`.
Watch out for a nested `sd-webui-anima-dc/sd-webui-anima-dc` folder when extracting a zip.

## Models

| File | Download | Put it in |
|---|---|---|
| `Battera-Anima-DC-v1.0.safetensors` | [Battera-Anima-DC](https://huggingface.co/FA770/Battera-Anima-DC) | `models/Stable-diffusion` |
| `Battera-VAE-f32c128-v1.0.safetensors` | [Battera-VAE-f32c128](https://huggingface.co/FA770/Battera-VAE-f32c128) | `models/VAE` |
| Qwen3 0.6B (same as stock Anima) | | `models/text_encoder` |

The model and the VAE are a matched pair. Use the same version number for both.

## Usage

1. **Checkpoint**: `Battera-Anima-DC-v1.0`
2. **VAE / Text Encoder**: select **both** `Battera-VAE-f32c128-v1.0` and Qwen3 0.6B
3. Width and height: multiples of 32 (other sizes are rounded down)
4. Other settings as for stock Anima; see the model card for recommendations

When the model loads, the console shows:

```
[Anima-DC] transformer: C=128 f32p1 ...
[Anima-DC] VAE: latent 128ch f32
```

txt2img, img2img and Hires. fix (Latent) work. Switching to other models is not affected; the
extension only acts on Anima-DC model files.

## Notes

- The VAE runs in Forge's default VAE precision (bf16 on supported GPUs). `--fp32-vae` and
  `--bf16-vae` work as usual. Do not quantize the VAE to FP4.
- Live previews in RGB mode are a rough approximation at latent resolution. Use `Full` for an
  exact preview.
- Anima's reference feature and ControlNet are not supported.

## How it works

Forge Neo's Anima support assumes a 16-channel latent, patch size 2 and the Qwen-Image VAE. The
transformer itself is the same, so this extension hooks four points in the loader and keeps the
rest of Forge's Anima code:

- model detection reads the input / output channels and patch size from the weights
- an Anima-DC VAE selected under **VAE / Text Encoder** is loaded as the VAE (encoder + decoder)
- the per-channel latent normalization stored in the model file is applied as the latent format
- the model runs with 2D latents and a 32× VAE

No WebUI files are modified. If a WebUI update breaks the extension, disabling it restores the
original behavior.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.

`lib_anima_dc/dcae.py` is a modified port of the DC-AE building blocks from
[mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) (Apache-2.0), obtained
through [lavinal712/AutoencoderKL](https://github.com/lavinal712/AutoencoderKL) (MIT). The
original license texts are in `licenses/`.

Forge Neo (AGPL-3.0) is imported at runtime; no WebUI source code is included.

Model weights are not part of this repository and have their own licenses. Battera-Anima-DC is a
derivative of Anima under the CircleStone Labs Non-Commercial License.

## Credits

The code in this repository was written by Claude (Anthropic) under the direction of FA770.

- GitHub: [FA7770](https://github.com/FA7770) / Hugging Face: [FA770](https://huggingface.co/FA770)
