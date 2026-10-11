# -*- coding: utf-8 -*-
# Copyright 2026 FA770
# SPDX-License-Identifier: Apache-2.0
#
# This file is a modified port of the DC-AE implementation from EfficientViT
# (https://github.com/mit-han-lab/efficientvit, Copyright 2023 Han Cai, Apache-2.0),
# obtained through https://github.com/lavinal712/AutoencoderKL (branch dc-ae, MIT).
# Changes: merged into one file, inference only, GSC shortcuts and per-stage
# norm / act in the decoder, TritonRMSNorm2d reimplemented in plain PyTorch,
# omegaconf / triton / unused blocks removed. See NOTICE and licenses/.
"""
DC-AE (Encoder + Decoder) の最小実装 (推論専用 / 依存は torch のみ)

元: sgm/modules/efficientvitmodules/model.py の Encoder / Decoder と nn/ops.py, nn/norm.py, nn/act.py。
state_dict のキー名が元実装と完全に一致するよう、モジュールの属性名と登録順を揃えてある
(ConvLayer.conv/norm/act, ResBlock.conv1/conv2, ResidualBlock.main/shortcut, OpSequential.op_list ...)。

相違点:
  * trms2d (TritonRMSNorm2d) を純 torch で実装。チャネル方向 (dim=1) の RMSNorm + affine。
    tools/extract_decoder.py --verify で元実装との一致を確認できる
  * omegaconf / triton / gradient checkpointing は持たない
"""

from dataclasses import dataclass, field, fields
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# norm / act
# ---------------------------------------------------------------------------
class RMSNorm2d(nn.LayerNorm):
    """TritonRMSNorm2d の純 torch 版。(B,C,H,W) をチャネル方向で正規化する。

    nn.LayerNorm を継承するのは weight / bias のキー名と初期化を元実装と揃えるため。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        r = torch.rsqrt(xf.pow(2).mean(dim=1, keepdim=True) + self.eps)
        y = xf * r
        if self.elementwise_affine:
            y = y * self.weight.float().view(1, -1, 1, 1)
            if self.bias is not None:
                y = y + self.bias.float().view(1, -1, 1, 1)
        return y.to(x.dtype)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


def build_norm(name="bn2d", num_features=None) -> Optional[nn.Module]:
    if name == "bn2d":
        return nn.BatchNorm2d(num_features)
    if name == "ln":
        return nn.LayerNorm(num_features)
    if name == "ln2d":
        return LayerNorm2d(num_features)
    if name == "trms2d":
        return RMSNorm2d(num_features)
    return None


def build_act(name) -> Optional[nn.Module]:
    return {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "hswish": nn.Hardswish,
        "silu": nn.SiLU,
        "gelu": lambda: nn.GELU(approximate="tanh"),
    }.get(name, lambda: None)()


def _val2tuple(x, n):
    if isinstance(x, (list, tuple)):
        x = list(x)
        if len(x) == n:
            return tuple(x)
        return tuple(x + [x[-1]] * (n - len(x)))
    return tuple([x] * n)


# ---------------------------------------------------------------------------
# ops (Decoder で使うものだけ)
# ---------------------------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 groups=1, use_bias=False, dropout=0, norm="bn2d", act_func="relu"):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                              stride=(stride, stride), padding=padding,
                              dilation=(dilation, dilation), groups=groups, bias=use_bias)
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, factor):
        super().__init__()
        self.factor = factor
        self.conv = ConvLayer(in_channels, out_channels * factor ** 2, kernel_size,
                              use_bias=True, norm=None, act_func=None)

    def forward(self, x):
        return F.pixel_shuffle(self.conv(x), self.factor)


class InterpolateConvUpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, factor, mode="nearest"):
        super().__init__()
        self.factor, self.mode = factor, mode
        self.conv = ConvLayer(in_channels, out_channels, kernel_size,
                              use_bias=True, norm=None, act_func=None)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=self.factor, mode=self.mode))


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, factor):
        super().__init__()
        self.in_channels, self.out_channels, self.factor = in_channels, out_channels, factor
        assert out_channels * factor ** 2 % in_channels == 0
        self.repeats = out_channels * factor ** 2 // in_channels

    def forward(self, x):
        x = x.repeat_interleave(self.repeats, dim=1)
        return F.pixel_shuffle(x, self.factor)


class IdentityLayer(nn.Module):
    def forward(self, x):
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, mid_channels=None,
                 expand_ratio=1, use_bias=False, norm=("bn2d", "bn2d"), act_func=("relu6", None)):
        super().__init__()
        use_bias = _val2tuple(use_bias, 2)
        norm = _val2tuple(norm, 2)
        act_func = _val2tuple(act_func, 2)
        mid = round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        self.conv1 = ConvLayer(in_channels, mid, kernel_size, stride,
                               use_bias=use_bias[0], norm=norm[0], act_func=act_func[0])
        self.conv2 = ConvLayer(mid, out_channels, kernel_size, 1,
                               use_bias=use_bias[1], norm=norm[1], act_func=act_func[1])

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ResidualBlock(nn.Module):
    def __init__(self, main, shortcut, post_act=None, pre_norm=None):
        super().__init__()
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward(self, x):
        if self.main is None:
            return x
        m = self.main(x if self.pre_norm is None else self.pre_norm(x))
        if self.shortcut is None:
            return m
        res = m + self.shortcut(x)
        if self.post_act:
            res = self.post_act(res)
        return res


class OpSequential(nn.Module):
    def __init__(self, op_list):
        super().__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x):
        for op in self.op_list:
            x = op(x)
        return x


class C2SChannelDuplicating(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2):
        super().__init__()
        self.factor = factor
        self.target_c = out_channels * factor ** 2
        if self.target_c >= in_channels:
            assert self.target_c % in_channels == 0
            self.repeat, self.mode = self.target_c // in_channels, "repeat"
        else:
            assert in_channels % self.target_c == 0
            self.group, self.mode = in_channels // self.target_c, "average"

    def forward(self, x):
        if self.mode == "repeat":
            x = x.repeat_interleave(self.repeat, dim=1)
        else:
            B, C, H, W = x.shape
            x = x.view(B, self.target_c, self.group, H, W).mean(dim=2)
        return F.pixel_shuffle(x, self.factor)


# ---------------------------------------------------------------------------
# config / builders
# ---------------------------------------------------------------------------
@dataclass
class DecoderConfig:
    in_channels: int = 3
    latent_channels: int = 128
    in_shortcut: Optional[str] = "duplicating"
    width_list: tuple = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: Any = "trms2d"
    act: Any = "silu"
    upsample_block_type: str = "ConvPixelShuffle"
    upsample_match_channel: bool = True
    upsample_shortcut: Optional[str] = "duplicating"
    out_norm: str = "trms2d"
    out_act: str = "relu"

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**{k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()})

    def to_dict(self):
        return {f.name: (list(getattr(self, f.name)) if isinstance(getattr(self, f.name), (list, tuple))
                         else getattr(self, f.name)) for f in fields(self)}


def build_block(block_type, in_channels, out_channels, norm, act):
    if block_type == "ResBlock":
        assert in_channels == out_channels
        main = ResBlock(in_channels, out_channels, kernel_size=3, stride=1,
                        use_bias=(True, False), norm=(None, norm), act_func=(act, None))
        return ResidualBlock(main, IdentityLayer())
    raise ValueError(f"block_type {block_type} is not supported (ResBlock only)")


def build_stage_main(width, depth, block_type, norm, act, input_width):
    stage = []
    for d in range(depth):
        bt = block_type[d] if isinstance(block_type, list) else block_type
        stage.append(build_block(bt, width if d > 0 else input_width, width, norm, act))
    return stage


def build_upsample_block(block_type, in_channels, out_channels, shortcut):
    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(in_channels, out_channels, kernel_size=3, factor=2)
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(in_channels, out_channels, kernel_size=3, factor=2)
    else:
        raise ValueError(block_type)
    if shortcut is None:
        return block
    if shortcut == "duplicating":
        return ResidualBlock(block, ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels, out_channels, factor=2))
    raise ValueError(shortcut)


def build_decoder_project_in_block(in_channels, out_channels, shortcut):
    block = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,
                      use_bias=True, norm=None, act_func=None)
    if shortcut is None:
        return block
    if shortcut == "duplicating":
        return ResidualBlock(block, ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels, out_channels, factor=1))
    raise ValueError(shortcut)


def build_decoder_project_out_block(in_channels, out_channels, factor, upsample_block_type, norm, act):
    layers = [build_norm(norm, in_channels), build_act(act)]
    if factor == 1:
        layers.append(ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,
                                use_bias=True, norm=None, act_func=None))
    elif factor == 2:
        layers.append(build_upsample_block(upsample_block_type, in_channels, out_channels, None))
    else:
        raise ValueError(factor)
    return OpSequential(layers)


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg
        n = len(cfg.width_list)
        self.num_stages = n
        assert len(cfg.depth_list) == n
        block_type = list(cfg.block_type) if isinstance(cfg.block_type, (list, tuple)) else cfg.block_type
        norm = list(cfg.norm) if isinstance(cfg.norm, (list, tuple)) else cfg.norm
        act = list(cfg.act) if isinstance(cfg.act, (list, tuple)) else cfg.act

        self.project_in = build_decoder_project_in_block(
            cfg.latent_channels, cfg.width_list[-1], cfg.in_shortcut)

        self.use_gsc = cfg.upsample_shortcut == "gsc"
        gsc_list = []
        stages = []
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage = []
            if stage_id < n - 1 and depth > 0:
                in_ch = cfg.width_list[stage_id + 1]
                out_ch = width if cfg.upsample_match_channel else cfg.width_list[stage_id + 1]
                up_sc = None if self.use_gsc else cfg.upsample_shortcut
                stage.append(build_upsample_block(cfg.upsample_block_type, in_ch, out_ch, up_sc))
                gsc_list.insert(0, C2SChannelDuplicating(in_ch, width, 2) if self.use_gsc else None)
            else:
                gsc_list.insert(0, None)
            bt = block_type[stage_id] if isinstance(block_type, list) else block_type
            nm = norm[stage_id] if isinstance(norm, list) else norm
            ac = act[stage_id] if isinstance(act, list) else act
            stage.extend(build_stage_main(
                width, depth, bt, nm, ac,
                input_width=width if cfg.upsample_match_channel
                else cfg.width_list[min(stage_id + 1, n - 1)]))
            stages.insert(0, OpSequential(stage))
        self.stages = nn.ModuleList(stages)
        self.gsc_shortcuts = nn.ModuleList([m if m is not None else nn.Identity() for m in gsc_list])
        self._gsc_active = [m is not None for m in gsc_list]

        self.project_out = build_decoder_project_out_block(
            cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            cfg.in_channels,
            1 if cfg.depth_list[0] > 0 else 2,
            cfg.upsample_block_type, cfg.out_norm, cfg.out_act)

    @property
    def spatial_compression_ratio(self) -> int:
        # depth=0 の段も project_out の factor 2 で補われるので 2^(段数-1)
        return 2 ** (self.num_stages - 1)

    def forward(self, x):
        x = self.project_in(x)
        for i in range(len(self.stages) - 1, -1, -1):
            stage = self.stages[i]
            if len(stage.op_list) == 0:
                continue
            if self._gsc_active[i]:
                x_in = x
                x = stage(x) + self.gsc_shortcuts[i](x_in)
            else:
                x = stage(x)
        return self.project_out(x)


CONFIG_META_KEY = "dcgen_decoder_config"


# ===========================================================================
# Encoder
# ===========================================================================
class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, factor):
        super().__init__()
        self.factor = factor
        assert out_channels % factor ** 2 == 0
        self.conv = ConvLayer(in_channels, out_channels // factor ** 2, kernel_size,
                              use_bias=True, norm=None, act_func=None)

    def forward(self, x):
        return F.pixel_unshuffle(self.conv(x), self.factor)


class PixelUnshuffleConvDownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, factor=2):
        super().__init__()
        self.factor = factor
        self.conv = ConvLayer(in_channels * factor ** 2, out_channels, kernel_size,
                              use_bias=True, norm=None, act_func=None)

    def forward(self, x):
        return self.conv(F.pixel_unshuffle(x, self.factor))


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, factor):
        super().__init__()
        self.in_channels, self.out_channels, self.factor = in_channels, out_channels, factor
        assert in_channels * factor ** 2 % out_channels == 0
        self.group_size = in_channels * factor ** 2 // out_channels

    def forward(self, x):
        x = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = x.shape
        return x.view(B, self.out_channels, self.group_size, H, W).mean(dim=2)


# GSC 用。演算は上と同じだが元実装で別クラスなので揃えておく
S2CChannelAveraging = PixelUnshuffleChannelAveragingDownSampleLayer


@dataclass
class EncoderConfig:
    in_channels: int = 3
    latent_channels: int = 128
    width_list: tuple = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple = (2, 2, 2, 2, 2, 2)
    block_type: Any = "ResBlock"
    norm: str = "trms2d"
    act: str = "silu"
    downsample_block_type: str = "ConvPixelUnshuffle"
    downsample_match_channel: bool = True
    downsample_shortcut: Optional[str] = "averaging"
    out_norm: Optional[str] = None
    out_act: Optional[str] = None
    out_shortcut: Optional[str] = "averaging"
    double_latent: bool = False

    from_dict = DecoderConfig.__dict__["from_dict"]
    to_dict = DecoderConfig.to_dict


def build_downsample_block(block_type, in_channels, out_channels, shortcut):
    if block_type == "Conv":
        block = ConvLayer(in_channels, out_channels, kernel_size=3, stride=2,
                          use_bias=True, norm=None, act_func=None)
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(in_channels, out_channels, kernel_size=3, factor=2)
    elif block_type == "PixelUnshuffleConv":
        block = PixelUnshuffleConvDownSampleLayer(in_channels, out_channels, kernel_size=1, factor=2)
    else:
        raise ValueError(block_type)
    if shortcut is None:
        return block
    if shortcut == "averaging":
        return ResidualBlock(block, PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels, out_channels, factor=2))
    raise ValueError(shortcut)


def build_encoder_project_in_block(in_channels, out_channels, factor, downsample_block_type):
    if factor == 1:
        return ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,
                         use_bias=True, norm=None, act_func=None)
    if factor == 2:
        return build_downsample_block(downsample_block_type, in_channels, out_channels, None)
    raise ValueError(factor)


def build_encoder_project_out_block(in_channels, out_channels, norm, act, shortcut):
    block = OpSequential([
        build_norm(norm, in_channels),
        build_act(act),
        ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,
                  use_bias=True, norm=None, act_func=None),
    ])
    if shortcut is None:
        return block
    if shortcut == "averaging":
        return ResidualBlock(block, PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels, out_channels, factor=1))
    raise ValueError(shortcut)


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        n = len(cfg.width_list)
        self.num_stages = n
        assert len(cfg.depth_list) == n
        block_type = list(cfg.block_type) if isinstance(cfg.block_type, (list, tuple)) else cfg.block_type

        self.project_in = build_encoder_project_in_block(
            cfg.in_channels,
            cfg.width_list[0] if cfg.depth_list[0] > 0 else cfg.width_list[1],
            1 if cfg.depth_list[0] > 0 else 2,
            cfg.downsample_block_type)

        self.use_gsc = cfg.downsample_shortcut == "gsc"
        gsc_list, stages = [], []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            bt = block_type[stage_id] if isinstance(block_type, list) else block_type
            stage = build_stage_main(width, depth, bt, cfg.norm, cfg.act, input_width=width)
            if stage_id < n - 1 and depth > 0:
                out_ch = cfg.width_list[stage_id + 1] if cfg.downsample_match_channel else width
                stage.append(build_downsample_block(
                    cfg.downsample_block_type, width, out_ch,
                    None if self.use_gsc else cfg.downsample_shortcut))
                gsc_list.append(S2CChannelAveraging(width, out_ch, 2) if self.use_gsc else None)
            else:
                gsc_list.append(None)
            stages.append(OpSequential(stage))
        self.stages = nn.ModuleList(stages)
        self.gsc_shortcuts = nn.ModuleList([m if m is not None else nn.Identity() for m in gsc_list])
        self._gsc_active = [m is not None for m in gsc_list]

        self.project_out = build_encoder_project_out_block(
            cfg.width_list[-1],
            2 * cfg.latent_channels if cfg.double_latent else cfg.latent_channels,
            cfg.out_norm, cfg.out_act, cfg.out_shortcut)

    @property
    def spatial_compression_ratio(self) -> int:
        return 2 ** (self.num_stages - 1)

    def forward(self, x):
        x = self.project_in(x)
        for stage, gsc, active in zip(self.stages, self.gsc_shortcuts, self._gsc_active):
            if len(stage.op_list) == 0:
                continue
            x = stage(x) + gsc(x) if active else stage(x)
        return self.project_out(x)


ENCODER_META_KEY = "dcgen_encoder_config"
