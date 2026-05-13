import torch
from torch import nn
from timm.models.layers import to_2tuple
import torch.nn.functional as F
from numbers import Number
from src.utils.unettool2 import UpsampleDeterministic
def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer
        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.GroupNorm(32,out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))




class TransLayer(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        # self.c5_down = nn.Sequential(
        #     ConvBNReLU(768, out_c, 3, 1, 1),
        #     # last_module(in_dim=2048, out_dim=out_c),
        # )
        self.c4_down = nn.Sequential(ConvBNReLU(768, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(384, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(192, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(96, out_c, 3, 1, 1))

    def forward(self, c1,c2,c3,c4):
        # assert isinstance(xs, (tuple, list))
        # assert len(xs) == 5
        # c1, c2, c3, c4, c5 = xs
        # c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return  c1,c2,c3,c4
class TransLayer2(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        # self.c5_down = nn.Sequential(
        #     ConvBNReLU(768, out_c, 3, 1, 1),
        #     # last_module(in_dim=2048, out_dim=out_c),
        # )
        self.c4_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(320, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(128, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

    def forward(self, c1,c2,c3,c4):
        # assert isinstance(xs, (tuple, list))
        # assert len(xs) == 5
        # c1, c2, c3, c4, c5 = xs
        # c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return  c1,c2,c3,c4
class TransLayergai(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        # self.c5_down = nn.Sequential(
        #     ConvBNReLU(768, out_c, 3, 1, 1),
        #     # last_module(in_dim=2048, out_dim=out_c),
        # )
        self.c4_down = nn.Sequential(ConvBNReLU(768, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(384, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(192, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(96, out_c, 3, 1, 1))

    def forward(self, c1,c2,c3,c4):
        # assert isinstance(xs, (tuple, list))
        # assert len(xs) == 5
        # c1, c2, c3, c4, c5 = xs
        # c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return  c1,c2,c3,c4


class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ASPP, self).__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
        self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
        self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))


class TransLayeraspp(nn.Module):
    def __init__(self, out_c, last_module=ASPP):
        super().__init__()
        self.c4_down = nn.Sequential(
            # ConvBNReLU(2048, 256, 3, 1, 1),
            last_module(in_dim=768, out_dim=out_c),
        )
        # self.c4_down = nn.Sequential(ConvBNReLU(768, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(384, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(192, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(96, out_c, 3, 1, 1))

    def forward(self, c1, c2, c3, c4):
        # assert isinstance(xs, (tuple, list))
        # assert len(xs) == 5
        # c1, c2, c3, c4, c5 = xs
        # c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return c1, c2, c3, c4




class SIU(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        # self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_l = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        # self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        # self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 3, 1),
        )

    def forward(self, l, m, s, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        tgt_size = m.shape[2:]
        # 尺度缩小
        l = self.conv_l(l)
        # l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)

        # 尺度不变
        m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        s = self.conv_s(s)
        # s = cus_sample(s, mode="size", factors=m.shape[2:])
        # s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms

class SIU2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.dup = UpsampleDeterministic()
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 3, 1),
        )

    def forward(self, s, m, l, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        tgt_size = m.shape[2:]
        # 尺度缩小
        l = self.conv_l_pre_down(l)

        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)
        # 尺度不变
        m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        # s = self.conv_s(s)
        s=self.conv_s_pre_up(s)
        # s = cus_sample(s, mode="size", factors=m.shape[2:])
        s = self.dup(s)
        s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms

class SIU3(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        # self.dup = UpsampleDeterministic()
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 3, 1),
        )

    def forward(self, s, m, l, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        tgt_size = m.shape[2:]
        # 尺度缩小
        l = self.conv_l_pre_down(l)

        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)
        # 尺度不变
        m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        # s = self.conv_s(s)
        s=self.conv_s_pre_up(s)
        s = cus_sample(s, mode="size", factors=m.shape[2:])
        # s = self.dup(s)
        s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms





class SIU4(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        # self.dup = UpsampleDeterministic()
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 3, 1),
        )

    def forward(self, s, m, l, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        # tgt_size = m.shape[2:]
        # 尺度缩小
        # l = self.conv_l_pre_down(l)

        # l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        # l = self.conv_l_post_down(l)
        # 尺度不变
        # m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        # s = self.conv_s(s)
        # s=self.conv_s_pre_up(s)
        # s = cus_sample(s, mode="size", factors=m.shape[2:])
        # s = self.dup(s)
        # s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms



class SIU5(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        # self.dup = UpsampleDeterministic()
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 3, 1),
        )

    def forward(self, s, m, l, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        # tgt_size = m.shape[2:]
        # 尺度缩小
        # l = self.conv_l_pre_down(l)

        # l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        # l = self.conv_l_post_down(l)
        # 尺度不变
        # m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        # s = self.conv_s(s)
        # s=self.conv_s_pre_up(s)
        # s = cus_sample(s, mode="size", factors=m.shape[2:])
        # s = self.dup(s)
        # s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms


class HMU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []

        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(3, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(2, dim=1))

        out = torch.cat([o[0] for o in outs], dim=1)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)



def cus_sample(
    feat: torch.Tensor,
    mode=None,
    factors=None,
    *,
    interpolation="bilinear",
    align_corners=False,
) -> torch.Tensor:
    """
    :param feat: 输入特征
    :param mode: size/scale
    :param factors: shape list for mode=size or scale list for mode=scale
    :param interpolation:
    :param align_corners: 具体差异可见https://www.yuque.com/lart/idh721/ugwn46
    :return: the resized tensor
    """
    if mode is None:
        return feat
    else:
        if factors is None:
            raise ValueError(
                f"factors should be valid data when mode is not None, but it is {factors} now."
                f"feat.shape: {feat.shape}, mode: {mode}, interpolation: {interpolation}, align_corners: {align_corners}"
            )

    interp_cfg = {}
    if mode == "size":
        if isinstance(factors, Number):
            factors = (factors, factors)
        assert isinstance(factors, (list, tuple)) and len(factors) == 2
        factors = [int(x) for x in factors]
        if factors == list(feat.shape[2:]):
            return feat
        interp_cfg["size"] = factors
    elif mode == "scale":
        assert isinstance(factors, (int, float))
        if factors == 1:
            return feat
        recompute_scale_factor = None
        if isinstance(factors, float):
            recompute_scale_factor = False
        interp_cfg["scale_factor"] = factors
        interp_cfg["recompute_scale_factor"] = recompute_scale_factor
    else:
        raise NotImplementedError(f"mode can not be {mode}")

    if interpolation == "nearest":
        if align_corners is False:
            align_corners = None
        assert align_corners is None, (
            "align_corners option can only be set with the interpolating modes: "
            "linear | bilinear | bicubic | trilinear, so we will set it to None"
        )
    try:
        result = F.interpolate(feat, mode=interpolation, align_corners=align_corners, **interp_cfg)
    except NotImplementedError as e:
        print(
            f"shape: {feat.shape}\n"
            f"mode={mode}\n"
            f"factors={factors}\n"
            f"interpolation={interpolation}\n"
            f"align_corners={align_corners}"
        )
        raise e
    except Exception as e:
        raise e
    return result
