# Copyright 2026 FA770
# SPDX-License-Identifier: Apache-2.0
"""
Forge Neo のローダーに掛けるフック。

Neo の Anima 対応は「in=16ch / patch=2 / Qwen-Image VAE (f8, 5D latent)」を前提にしている。
DC-Gen 版は transformer の中身 (28 ブロック + LLM adapter) が同じで、入出力層と AE だけが違う。
そこで 4 か所だけ差し替える:

  1. huggingface_guess.detection.detect_unet_config
       ファイルに dcgen.latents_std があれば、in / out / patch を実際の形から決める
  2. backend.loader.replace_state_dict
       「VAE / Text Encoder」に指定された DC-AE (extract_ae.py の出力) を VAE として受け取る
  3. backend.loader.split_state_dict
       dcgen.* を transformer から外し、専用の config クラスと latent format に切り替える
  4. backend.loader.load_huggingface_component
       VAE として DC-AE (encoder + decoder) を組む

エンジン (AnimaDCEngine) は Anima を継承し、VAE を f32 / 2D latent として扱う。
transformer は (B,C,T,H,W) を要求するので、forward の前後で T 軸を足し引きする。
"""

import json
import math
from types import SimpleNamespace

import torch
import torch.nn as nn

from .dcae import (CONFIG_META_KEY, ENCODER_META_KEY, Decoder, DecoderConfig, Encoder,
                   EncoderConfig)



class _Log:
    """Forge のコンソールは root logger の INFO を出さないので print する。"""

    @staticmethod
    def info(msg):
        print(msg, flush=True)

    warning = info


log = _Log()

MARK_STD = "dcgen.latents_std"
MARK_MEAN = "dcgen.latents_mean"
EMB_KEY = "x_embedder.proj.1.weight"
HEAD_PREFIX = "final_layer.head."
AE_SIGNATURE = "decoder.project_in."

_installed = False


def _read_metadata(path) -> dict:
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return f.metadata() or {}
    except Exception as e:
        log.warning(f"[Anima-DC] cannot read metadata ({path}): {e}")
        return {}


# ===========================================================================
# latent format
# ===========================================================================
def _make_latent_format(std, mean, new_f, new_patch):
    from huggingface_guess import latent

    class AnimaDCLatentFormat(latent.LatentFormat):
        """process_in = (z - mean) / std,  process_out = z * std + mean"""
        taesd_decoder_name = None

        def __init__(self):
            self.scale_factor = 1.0
            self.latent_channels = std.numel()
            self.spacial_downscale_ratio = new_f
            self.new_f, self.new_patch = new_f, new_patch
            self.latents_std = std.float().view(1, -1, 1, 1)
            self.latents_mean = (mean.float().view(1, -1, 1, 1) if mean is not None
                                 else torch.zeros_like(self.latents_std))
            # ライブプレビュー (RGB 近似) 用。AE を読んだあと fit_preview_factors が埋める
            self.latent_rgb_factors = [[0.0, 0.0, 0.0] for _ in range(self.latent_channels)]
            self.latent_rgb_factors_bias = [0.0, 0.0, 0.0]

        def _stats(self, x):
            s = self.latents_std.to(x.device, x.dtype)
            m = self.latents_mean.to(x.device, x.dtype)
            if x.ndim == 5:
                s, m = s.unsqueeze(2), m.unsqueeze(2)
            return s, m

        def process_in(self, x):
            s, m = self._stats(x)
            return (x - m) / s

        def process_out(self, x):
            s, m = self._stats(x)
            return x * s + m

    return AnimaDCLatentFormat()


# ===========================================================================
# AE
# ===========================================================================
class AnimaDCAutoencoder(nn.Module):
    """Forge の VAE ラッパーが期待する形 (encode / decode / process_in / process_out / config)。"""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.f = decoder.spatial_compression_ratio
        self.config = SimpleNamespace(latent_channels=decoder.cfg.latent_channels)
        self.latent_format = None      # ForgeDiffusionEngine.__init__ が入れる

    def _sync(self, x):
        # Forge 側のモデル管理が移してくれるはずだが、素の nn.Module なので念のため合わせる
        p = next(self.parameters())
        if p.device != x.device or p.dtype != x.dtype:
            self.to(device=x.device, dtype=x.dtype)

    def encode(self, x):
        """(B,3,H,W) [-1,1] -> 生の潜在 (B,C,H/f,W/f)。f の倍数に中央クロップする。"""
        self._sync(x)
        H, W = x.shape[-2:]
        h, w = H // self.f * self.f, W // self.f * self.f
        if h == 0 or w == 0:
            raise RuntimeError(f"Image too small ({W}x{H}, minimum {self.f})")
        t, l = (H - h) // 2, (W - w) // 2
        return self.encoder(x[..., t:t + h, l:l + w])

    def decode(self, z):
        self._sync(z)
        return self.decoder(z)

    def process_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_out(self, latent):
        return self.latent_format.process_out(latent)


def _build_ae(guess, state_dict):
    meta = getattr(guess, "dcgen_ae_meta", None)
    if not state_dict or meta is None:
        raise RuntimeError(
            "Anima-DC models need their own VAE. Add it (e.g. Battera-VAE-f32c128) to 'VAE / Text Encoder'. "
            "The Qwen-Image VAE does not work with this model")
    if CONFIG_META_KEY not in meta or ENCODER_META_KEY not in meta:
        raise RuntimeError("This VAE file has no Anima-DC VAE metadata. Use an Anima-DC VAE file (e.g. Battera-VAE-f32c128)")

    def build(cls, cfg_cls, key, prefix):
        m = cls(cfg_cls.from_dict(json.loads(meta[key])))
        m.load_state_dict({k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)},
                          strict=True)
        return m

    enc = build(Encoder, EncoderConfig, ENCODER_META_KEY, "encoder.")
    dec = build(Decoder, DecoderConfig, CONFIG_META_KEY, "decoder.")
    C = guess.dcgen["channels"]
    if dec.cfg.latent_channels != C:
        raise RuntimeError(f"VAE latent has {dec.cfg.latent_channels} channels but the model expects {C}")
    if dec.spatial_compression_ratio != guess.dcgen["new_f"]:
        raise RuntimeError(f"VAE compression f{dec.spatial_compression_ratio} does not match the "
                           f"model (f{guess.dcgen['new_f']})")
    ae = AnimaDCAutoencoder(enc, dec).float().eval().requires_grad_(False)
    log.info(f"[Anima-DC] VAE: latent {C}ch f{ae.f}")
    return ae


@torch.inference_mode()
def fit_preview_factors(vae, fmt, size=384, n=8, seed=0):
    """ライブプレビューの RGB 近似用に、正規化済み潜在 -> RGB の線形写像を最小二乗で決める。

    低周波の合成画像を encoder に通し、潜在と (1/f に縮小した) 画像の対応から求める。
    色と大まかな明暗が分かる程度のもの。正確に見たいときは Live preview method を Full にする。
    """
    f = fmt.new_f
    g = torch.Generator(device="cpu").manual_seed(seed)
    imgs = []
    for i in range(n):
        k = (2, 3, 4, 6, 8, 12, 16, 24)[i % 8]
        low = torch.rand(1, 3, k, k, generator=g)
        up = torch.nn.functional.interpolate(low, size=(size, size), mode="bicubic", align_corners=False)
        imgs.append(up.clamp(0, 1))
    x = torch.cat(imgs, 0)                                   # (n,3,H,W) 0..1
    z = vae.encode(x.movedim(1, -1)).float().cpu()           # 生の潜在 (n,C,h,w)
    zn = fmt.process_in(z)
    tgt = torch.nn.functional.avg_pool2d(x, f) * 2.0 - 1.0   # (n,3,h,w)  [-1,1]
    C = zn.shape[1]
    A = zn.movedim(1, -1).reshape(-1, C).double()
    A = torch.cat([A, torch.ones(A.shape[0], 1, dtype=A.dtype)], 1)
    B = tgt.movedim(1, -1).reshape(-1, 3).double()
    lam = 1e-2 * A.shape[0]
    reg = torch.eye(C + 1, dtype=A.dtype) * lam
    reg[-1, -1] = 0.0
    W = torch.linalg.solve(A.T @ A + reg, A.T @ B)           # (C+1, 3)
    fmt.latent_rgb_factors = W[:C].float().tolist()
    fmt.latent_rgb_factors_bias = W[C].float().tolist()
    err = float(((A @ W - B) ** 2).mean().sqrt())
    log.info(f"[Anima-DC] live preview RGB factors fitted (rms error {err:.3f})")


# ===========================================================================
# transformer
# ===========================================================================
def _wrap_forward(model):
    """Anima は (B,C,T,H,W) を取る。2D latent のまま扱えるよう T 軸を足し引きする。"""
    if getattr(model, "_dcgen_wrapped", False):
        return
    orig = model.forward

    def forward(x, *args, **kwargs):
        if x.ndim == 4:
            return orig(x.unsqueeze(2), *args, **kwargs).squeeze(2)
        return orig(x, *args, **kwargs)

    model.forward = forward
    model._dcgen_wrapped = True


def _apply_rope(model, info):
    for part in str(info["rope"]).split("+"):
        part = part.strip()
        if part in ("", "none"):
            continue
        if part == "ntk":
            pe = model.pos_embedder
            dim_h = pe.dim_spatial_range.numel() * 2
            ratio = (info["new_f"] * info["new_patch"]) // 16
            f = (4.0 / ratio) ** (dim_h / (dim_h - 2))
            pe.h_ntk_factor = pe.w_ntk_factor = f
            log.info(f"[Anima-DC] rope ntk factor={f:.4f}")
        else:
            raise RuntimeError(f"rope='{part}' is not supported. Use a model trained with rope=none")


# ===========================================================================
# hooks
# ===========================================================================
def install():
    global _installed
    if _installed:
        return

    import backend.loader as loader
    from backend import memory_management
    from backend.args import dynamic_args
    from backend.diffusion_engine.anima import Anima as AnimaEngine
    from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
    from backend.patcher.clip import CLIP
    from backend.patcher.unet import UnetPatcher
    from backend.patcher.vae import VAE
    from backend.text_processing.anima_engine import Qwen06Engine
    from huggingface_guess import detection, model_list

    # ---- 1. 判定 -----------------------------------------------------------
    orig_detect = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        k_std, k_emb = key_prefix + MARK_STD, key_prefix + EMB_KEY
        if k_std not in state_dict or k_emb not in state_dict:
            return orig_detect(state_dict, key_prefix, *args, **kwargs)
        C = state_dict[k_std].numel()
        D, n_in = state_dict[k_emb].shape
        p = math.isqrt(n_in // (C + 1))
        if p * p * (C + 1) != n_in:
            raise RuntimeError(f"x_embedder shape {(D, n_in)} does not match a {C}-channel latent")
        fake = dict(state_dict)
        fake[k_emb] = torch.empty((D, (16 + 1) * 4), device="meta")   # 元の Anima の形に見せる
        cfg = orig_detect(fake, key_prefix, *args, **kwargs)
        if cfg is None or cfg.get("image_model") != "anima":
            raise RuntimeError("Anima-DC model was not detected as Anima")
        cfg.update(in_channels=C, out_channels=C, patch_spatial=p)
        return cfg

    detection.detect_unet_config = detect_unet_config

    # ---- config クラス ------------------------------------------------------
    class AnimaDCConfig(model_list.Anima):
        pass

    # ---- 2. AE を VAE として受け取る ---------------------------------------
    orig_replace = loader.replace_state_dict

    def replace_state_dict(sd, asd, guess, path):
        if isinstance(asd, dict) and any(k.startswith(AE_SIGNATURE) for k in asd):
            prefix = guess.vae_key_prefix[0]
            for k in [k for k in sd if k.startswith(prefix)]:
                del sd[k]
            for k, v in asd.items():
                sd[prefix + k] = v
            guess.dcgen_ae_meta = _read_metadata(path)
            return sd
        return orig_replace(sd, asd, guess, path)

    loader.replace_state_dict = replace_state_dict

    # ---- 3. dcgen.* を外して config を切り替える ----------------------------
    orig_split = loader.split_state_dict

    def split_state_dict(path, additional_state_dicts=None):
        state_dict, guess = orig_split(path, additional_state_dicts=additional_state_dicts)
        dit = state_dict.get("transformer") or {}
        if MARK_STD not in dit:
            return state_dict, guess

        md = _read_metadata(path)
        std = dit.pop(MARK_STD)
        mean = dit.pop(MARK_MEAN, None)
        C = std.numel()
        new_patch = int(guess.unet_config["patch_spatial"])
        if "dcgen_new_f" not in md:
            raise RuntimeError("This file has no dcgen_* metadata. Use an Anima-DC model (e.g. Battera-Anima-DC)")
        new_f = int(md["dcgen_new_f"])
        head_type = md.get("dcgen_head_type", "linear")
        if head_type != "linear" or any(k.startswith(HEAD_PREFIX) for k in dit):
            raise NotImplementedError(
                f"head_type={head_type} is not supported in Forge. Use a model with a linear head")

        guess.__class__ = AnimaDCConfig
        guess.latent_format = _make_latent_format(std, mean, new_f, new_patch)
        guess.dcgen = {"channels": C, "new_f": new_f, "new_patch": new_patch,
                       "rope": md.get("dcgen_rope", "none")}
        log.info(f"[Anima-DC] transformer: C={C} f{new_f}p{new_patch} rope={guess.dcgen['rope']} "
                 f"ema={md.get('dcgen_ema')} lora={md.get('dcgen_lora_merged')}")
        return state_dict, guess

    loader.split_state_dict = split_state_dict

    # ---- 4. VAE の構築 -----------------------------------------------------
    orig_component = loader.load_huggingface_component

    def load_huggingface_component(guess, component_name, *args, **kwargs):
        if component_name == "vae" and hasattr(guess, "dcgen"):
            state_dict = args[-1] if args else kwargs.get("state_dict")
            return _build_ae(guess, state_dict)
        return orig_component(guess, component_name, *args, **kwargs)

    loader.load_huggingface_component = load_huggingface_component

    # ---- エンジン ----------------------------------------------------------
    class AnimaDCEngine(AnimaEngine):
        matched_guesses = [AnimaDCConfig]

        def __init__(self, estimated_config, huggingface_components):
            ForgeDiffusionEngine.__init__(self, estimated_config, huggingface_components)
            info = estimated_config.dcgen
            fmt = estimated_config.latent_format

            clip = CLIP(model_dict={"qwen3_06b": huggingface_components["text_encoder"]},
                        tokenizer_dict={"qwen3_06b": huggingface_components["tokenizer"],
                                        "t5xxl": huggingface_components["tokenizer_2"]})

            # dtype は Forge の既定 (対応 GPU なら bf16、それ以外は fp32)。--fp32-vae 等で変えられる
            vae = VAE(model=huggingface_components["vae"])
            f = info["new_f"]
            vae.upscale_ratio = vae.downscale_ratio = f
            vae.latent_channels = info["channels"]
            vae.memory_used_encode = lambda shape, dtype: 3200 * shape[2] * shape[3] * memory_management.dtype_size(dtype)
            vae.memory_used_decode = lambda shape, dtype: 3200 * shape[2] * shape[3] * (f * f) * memory_management.dtype_size(dtype)

            transformer = huggingface_components["transformer"]
            _wrap_forward(transformer)
            _apply_rope(transformer, info)

            unet = UnetPatcher.from_model(model=transformer, diffusers_scheduler=None,
                                          k_predictor=self._get_predictor(), config=estimated_config)

            self.text_processing_engine_qwen = Qwen06Engine(
                text_encoder=clip.cond_stage_model.qwen3_06b,
                qwen_tokenizer=clip.tokenizer.qwen3_06b,
                t5_tokenizer=clip.tokenizer.t5xxl,
            )

            self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
            self.forge_objects_original = self.forge_objects.shallow_copy()
            self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

            self.is_wan = False        # 2D latent (B,C,H,W)。Flux2 と同じ扱い
            self.use_shift = True

            try:
                fit_preview_factors(vae, fmt)
            except Exception as e:
                log.warning(f"[Anima-DC] could not fit live preview RGB factors (previews will be gray): {e}")

        @torch.inference_mode()
        def get_learned_conditioning(self, prompt):
            memory_management.load_model_gpu(self.forge_objects.clip.patcher)
            dynamic_args.ref_latents.clear()       # 参照 latent (Anima の reference 機能) は未対応
            return self.text_processing_engine_qwen(prompt)

        @torch.inference_mode()
        def encode_first_stage(self, x):
            return ForgeDiffusionEngine.encode_first_stage(self, x)

    loader.possible_models = (*loader.possible_models, AnimaDCEngine)

    _installed = True
    log.info("[Anima-DC] loader hooks installed")
