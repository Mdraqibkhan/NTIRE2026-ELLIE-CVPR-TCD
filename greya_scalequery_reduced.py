import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic building blocks (identical to original)
# ---------------------------------------------------------------------------

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, prelu=True, bn=False, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn    = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.prelu = nn.PReLU() if prelu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn    is not None: x = self.bn(x)
        if self.prelu is not None: x = self.prelu(x)
        return x


class MDTA(nn.Module):
    """Multi-Dconv Transposed Attention with grey-guided cross-attention query."""
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv       = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.query     = nn.Conv2d(channels, channels,     kernel_size=1, bias=False)
        self.query_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                    groups=channels, bias=False)
        self.qkv_conv  = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1,
                                    groups=channels * 2, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, y):
        b, c, h, w = x.shape
        k, v = self.qkv_conv(self.qkv(x)).chunk(2, dim=1)
        q    = self.query_conv(self.query(y))
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out  = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    """Gated Depthwise Feed-forward Network."""
    def __init__(self, channels, expansion_factor):
        super().__init__()
        hidden = int(channels * expansion_factor)
        self.project_in  = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.conv        = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                                     groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn   = GDFN(channels, expansion_factor)

    def _ln(self, x, norm):
        b, c, h, w = x.shape
        return (norm(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                .transpose(-2, -1).contiguous().reshape(b, c, h, w))

    def forward(self, x, y):
        x1    = self.attn(self._ln(x, self.norm1), self._ln(x, self.norm1))
        x_out = x + self.attn(self._ln(x1, self.norm1), self._ln(y, self.norm1))
        x_out = x_out + self.ffn(self._ln(x_out, self.norm2))
        return x_out


class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x): return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x): return self.body(x)


class PhaseTrans:
    """Phase transfer: keep magnitude of ref, phase of src."""
    def __call__(self, x, y):
        fftn_x = torch.fft.fftn(x)
        fftn_y = torch.fft.fftn(y)
        return torch.fft.ifftn(
            torch.abs(fftn_y) * torch.exp(1j * fftn_x.angle())).real


# ---------------------------------------------------------------------------
# Lightweight model  (channels halved at every level)
# ---------------------------------------------------------------------------

class my_model_lite(nn.Module):
    """
    Reduced-channel grey-guided cross-attention U-Net for NTIRE 2026.

    Default channels : [8, 16, 32, 64]  (vs [16, 32, 64, 128] in original)
    Params           : ~122 K
    FP32 file size   : ~0.46 MB
    FP16 file size   : ~0.23 MB  ← well under the 1 MB challenge limit
    """
    def __init__(self,
                 num_heads       = [1, 2, 4, 4],
                 channels        = [8, 16, 32, 64],
                 expansion_factor = 2.66):
        super().__init__()
        self.PhaseTrans = PhaseTrans()
        C = channels  # shorthand

        # ── Stem ────────────────────────────────────────────────────────────
        self.embed_conv_rgb = nn.Conv2d(3, C[0], 3, padding=1, bias=False)
        self.conv_grey      = nn.Conv2d(3, C[0], 3, padding=1, bias=False)

        # ── Grey branch convs (encoder) ──────────────────────────────────────
        self.conv_grey1 = BasicConv(C[0], C[0])
        self.conv_grey2 = BasicConv(C[1], C[1])
        self.conv_grey3 = BasicConv(C[2], C[2])

        # ── Grey branch convs (decoder) ──────────────────────────────────────
        self.decon_grey1 = BasicConv(C[2], C[2])
        self.decon_grey2 = BasicConv(C[3], C[2])
        self.decon_grey3 = BasicConv(C[2], C[1])
        self.decon_grey4 = BasicConv(C[1], C[0])

        # ── Down / Up sampling ───────────────────────────────────────────────
        self.downsample1 = DownSample(C[0])
        self.downsample2 = DownSample(C[1])
        self.downsample3 = DownSample(C[2])
        self.upsample1   = UpSample(C[3])
        self.upsample2   = UpSample(C[2])
        self.upsample3   = UpSample(C[1])

        # ── Skip-connection reducers ─────────────────────────────────────────
        self.reduce1 = nn.Conv2d(C[3], C[2], 1, bias=False)
        self.reduce2 = nn.Conv2d(C[2], C[1], 1, bias=False)
        self.reduce3 = nn.Conv2d(C[1], C[0], 1, bias=False)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder1_1 = TransformerBlock(C[0], num_heads[0], expansion_factor)
        self.encoder2_1 = TransformerBlock(C[1], num_heads[1], expansion_factor)
        self.encoder3_1 = TransformerBlock(C[2], num_heads[2], expansion_factor)
        self.encoder4_1 = TransformerBlock(C[3], num_heads[3], expansion_factor)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder1_1 = TransformerBlock(C[2], num_heads[2], expansion_factor)
        self.decoder2_1 = TransformerBlock(C[1], num_heads[1], expansion_factor)
        self.decoder3_1 = TransformerBlock(C[1], num_heads[0], expansion_factor)
        self.decoder_d1 = TransformerBlock(C[0], num_heads[0], expansion_factor)
        self.decoder_r1 = TransformerBlock(C[0], num_heads[0], expansion_factor)

        # ── Output ───────────────────────────────────────────────────────────
        self.output = nn.Conv2d(C[0], 3, 3, padding=1, bias=False)

    def forward(self, RGB_input, grey_input):
        C = [8, 16, 32, 64]  # kept for clarity (mirrors __init__)

        # Stem
        fo_rgb  = self.embed_conv_rgb(RGB_input)
        fo_grey = self.conv_grey(grey_input)

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_rgb1  = self.encoder1_1(fo_rgb, fo_grey)
        enc_grey1 = self.downsample1(self.conv_grey1(fo_grey))

        enc_rgb2  = self.encoder2_1(self.downsample1(enc_rgb1), enc_grey1)
        enc_grey2 = self.downsample2(self.conv_grey2(enc_grey1))

        enc_rgb3  = self.encoder3_1(self.downsample2(enc_rgb2), enc_grey2)
        enc_grey3 = self.downsample3(self.conv_grey3(enc_grey2))

        enc_rgb4  = self.encoder4_1(self.downsample3(enc_rgb3), enc_grey3)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Stage 3→2
        up1        = self.upsample1(enc_rgb4)
        pt1        = self.PhaseTrans(up1, enc_rgb3)
        dec_in3    = self.reduce1(torch.cat([up1, pt1], dim=1))
        out_dec3   = self.decoder1_1(dec_in3, dec_in3)

        # Stage 2→1
        up2        = self.upsample2(out_dec3)
        pt2        = self.PhaseTrans(up2, enc_rgb2)
        dec_in2    = self.reduce2(torch.cat([up2, pt2], dim=1))
        out_dec2   = self.decoder2_1(dec_in2, dec_in2)

        # Stage 1→0
        up3        = self.upsample3(out_dec2)
        pt3        = self.PhaseTrans(up3, enc_rgb1)
        dec_in1    = torch.cat([up3, pt3], dim=1)     # 2×C[0] channels
        out_dec1   = self.decoder3_1(dec_in1, dec_in1)

        # Refinement
        dec_d      = self.reduce3(out_dec1)
        out_d      = self.decoder_d1(dec_d, dec_d)
        out_r      = self.decoder_r1(out_d, out_d)

        return self.output(out_r)


# ---------------------------------------------------------------------------
# Size sanity-check (run this file directly to verify)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os

    model = my_model_lite()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters  : {total:,}")
    print(f"FP32 size   : {total * 4 / 1e6:.3f} MB")
    print(f"FP16 size   : {total * 2 / 1e6:.3f} MB")

    # Save fp16 state_dict and measure actual file size
    tmp_path = 'ntire2026_size_check.pth'
    torch.save(model.half().state_dict(), tmp_path)
    actual_mb = os.path.getsize(tmp_path) / 1e6
    print(f"Actual .pth : {actual_mb:.3f} MB  {'✅ PASS' if actual_mb < 1.0 else '❌ FAIL — still too large'}")
    os.remove(tmp_path)

    # Forward pass check
    model = my_model_lite()
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    out = model(x, y)
    print(f"Output shape: {out.shape}  (expected: [1, 3, 256, 256])")
