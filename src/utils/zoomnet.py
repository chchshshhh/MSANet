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
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
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



from src.utils.zuifusion import SAGate,SAGate2
class SIUonly2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        # self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        #
        # self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        #
        # # self.dup = UpsampleDeterministic()
        # self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        # self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.gate1=SAGate(64,64)
        self.gate2=SAGate(64,64)
        self.trans = nn.Sequential(
            ConvBNReLU(2 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim,2, 1),
        )

    def forward(self, s, m, l, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""

        sl=self.gate1(s,l)
        sm=self.gate2(s,m)

        # attn = self.trans(torch.cat([sl,sm], dim=1))
        # attn_sl, attn_sm = torch.softmax(attn, dim=1).chunk(2, dim=1)
        # lms = attn_sl * sl + attn_sm * sm

        # if return_feats:
            # return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return sl+sm






class SIUonly3(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        # self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        #
        # self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        #
        # # self.dup = UpsampleDeterministic()
        # self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        # self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.gate1=SAGate(64,64)
        self.gate2=SAGate(64,64)
        self.trans = nn.Sequential(
            ConvBNReLU(2 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim,2, 1),
        )

    def forward(self, s, m,l, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""

        attn = self.trans(torch.cat([s,m], dim=1))
        attn_s, attn_m = torch.softmax(attn, dim=1).chunk(2, dim=1)
        sm = attn_m * m + attn_s * s

        sl=self.gate1(sm,l)


        # attn = self.trans(torch.cat([sl,sm], dim=1))
        # attn_sl, attn_sm = torch.softmax(attn, dim=1).chunk(2, dim=1)
        # lms = attn_sl * sl + attn_sm * sm

        # if return_feats:
            # return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return sl




class SIUonly4(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        # self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        #
        # self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        #
        # # self.dup = UpsampleDeterministic()
        # self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        # self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.gate1=SAGate2(64,64)
        # self.gate2=SAGate(64,64)
        self.trans = nn.Sequential(
            ConvBNReLU(2 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim,2, 1),
        )

    def forward(self, s, m,l, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""

        attn = self.trans(torch.cat([s,m], dim=1))
        attn_s, attn_m = torch.softmax(attn, dim=1).chunk(2, dim=1)
        sm = attn_m * m + attn_s * s

        sl=self.gate1(sm,l)


        # attn = self.trans(torch.cat([sl,sm], dim=1))
        # attn_sl, attn_sm = torch.softmax(attn, dim=1).chunk(2, dim=1)
        # lms = attn_sl * sl + attn_sm * sm

        # if return_feats:
            # return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return sl







class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, out_dim=None, add=False, ratio=8):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.add = add
        if out_dim is None:
            out_dim = in_dim
        self.out_dim = out_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width*height).permute(0, 2, 1)  # B X C X(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width*height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width*height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, self.out_dim, width, height)

        if self.add:
            out = self.gamma*out + x
        else:
            out = self.gamma*out
        return out  # , attention


class CrossModalAttention(nn.Module):
    """ CMA attention Layer"""

    def __init__(self, in_dim, activation=None, ratio=8, cross_value=True):
        super(CrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.cross_value = cross_value

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        proj_query = self.query_conv(x).view(
            B, -1, H*W).permute(0, 2, 1)  # B , HW, C
        proj_key = self.key_conv(y).view(
            B, -1, H*W)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention = self.softmax(energy)  # BX (N) X (N)
        if self.cross_value:
            proj_value = self.value_conv(y).view(
                B, -1, H*W)  # B , C , HW
        else:
            proj_value = self.value_conv(x).view(
                B, -1, H*W)  # B , C , HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma*out + x

        if self.activation is not None:
            out = self.activation(out)

        return out  # , attention


class DualCrossModalAttention(nn.Module):
    """ Dual CMA attention Layer"""

    def __init__(self, in_dim, activation=None, size=16, ratio=8, ret_att=False):
        super(DualCrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.ret_att = ret_att

        # query conv
        self.key_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv_share = nn.Conv2d(
            in_channels=in_dim//ratio, out_channels=in_dim//ratio, kernel_size=1)

        self.linear1 = nn.Linear(size*size, size*size)
        self.linear2 = nn.Linear(size*size, size*size)

        # separated value conv
        self.value_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.value_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        def _get_att(a, b):
            proj_key1 = self.key_conv_share(self.key_conv1(a)).view(
                B, -1, H*W).permute(0, 2, 1)  # B, HW, C
            proj_key2 = self.key_conv_share(self.key_conv2(b)).view(
                B, -1, H*W)  # B X C x (*W*H)
            energy = torch.bmm(proj_key1, proj_key2)  # B, HW, HW

            attention1 = self.softmax(self.linear1(energy))
            attention2 = self.softmax(self.linear2(
                energy.permute(0, 2, 1)))  # BX (N) X (N)

            return attention1, attention2

        att_y_on_x, att_x_on_y = _get_att(x, y)
        proj_value_y_on_x = self.value_conv2(y).view(
            B, -1, H*W)  # B, C, HW
        out_y_on_x = torch.bmm(proj_value_y_on_x, att_y_on_x.permute(0, 2, 1))
        out_y_on_x = out_y_on_x.view(B, C, H, W)
        out_x = self.gamma1*out_y_on_x + x

        proj_value_x_on_y = self.value_conv1(x).view(
            B, -1, H*W)  # B , C , HW
        out_x_on_y = torch.bmm(proj_value_x_on_y, att_x_on_y.permute(0, 2, 1))
        out_x_on_y = out_x_on_y.view(B, C, H, W)
        out_y = self.gamma2*out_x_on_y + y

        if self.ret_att:
            return out_x, out_y, att_y_on_x, att_x_on_y

        return out_x, out_y  # , attention



class SIUonly5(nn.Module):
    def __init__(self, in_dim,size=96):
        super().__init__()
        # self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        # self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        #
        # self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        #
        # # self.dup = UpsampleDeterministic()
        # self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        # self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.trans = nn.Sequential(
            ConvBNReLU(2 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim,2, 1),
        )
        self.fusion=DualCrossModalAttention(in_dim,size=size)

        self.cross_conv = nn.Conv2d(64 * 2, 64, 1, padding=0)
    def forward(self, s, m,l, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""

        attn = self.trans(torch.cat([s,m], dim=1))
        attn_s, attn_m = torch.softmax(attn, dim=1).chunk(2, dim=1)
        sm = attn_m * m + attn_s * s

        sm,l=self.fusion(sm,l)
        out=self.cross_conv(torch.cat([sm,l],dim=1))


        # attn = self.trans(torch.cat([sl,sm], dim=1))
        # attn_sl, attn_sm = torch.softmax(attn, dim=1).chunk(2, dim=1)
        # lms = attn_sl * sl + attn_sm * sm

        # if return_feats:
            # return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return out


class CIM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CIM, self).__init__()

        act_fn = nn.ReLU(inplace=True)
        self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        self.reduc_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_dim), act_fn, )
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_dim), act_fn, )

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(out_dim), act_fn, )
        # self.layer_ful2 = nn.Sequential(nn.Conv2d(out_dim + out_dim // 2, out_dim, kernel_size=3, stride=1, padding=1),
        #                                 nn.BatchNorm2d(out_dim), act_fn, )

    def forward(self, rgb, depth):
        ################################
        x_rgb = self.reduc_1(rgb)
        x_dep = self.reduc_2(depth)

        x_rgb1 = self.layer_10(x_rgb)
        x_dep1 = self.layer_20(x_dep)

        rgb_w = nn.Sigmoid()(x_rgb1)
        dep_w = nn.Sigmoid()(x_dep1)

        ##
        x_rgb_w = x_rgb.mul(dep_w)
        x_dep_w = x_dep.mul(rgb_w)

        x_rgb_r = x_rgb_w + x_rgb
        x_dep_r = x_dep_w + x_dep

        ## fusion
        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)

        ful_mul = torch.mul(x_rgb_r, x_dep_r)
        x_in1 = torch.reshape(x_rgb_r, [x_rgb_r.shape[0], 1, x_rgb_r.shape[1], x_rgb_r.shape[2], x_rgb_r.shape[3]])
        x_in2 = torch.reshape(x_dep_r, [x_dep_r.shape[0], 1, x_dep_r.shape[1], x_dep_r.shape[2], x_dep_r.shape[3]])
        x_cat = torch.cat((x_in1, x_in2), dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul, ful_max), dim=1)

        out1 = self.layer_ful1(ful_out)
        # out2 = self.layer_ful2(torch.cat([out1, xx], dim=1))

        return out1

class SIUonly6(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        # self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        #
        # self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        #
        # # self.dup = UpsampleDeterministic()
        # self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        # self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        # self.trans = nn.Sequential(
        #     ConvBNReLU(2 * in_dim, in_dim, 1),
        #
        #     ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        #     nn.Conv2d(in_dim,2, 1),
        # )
        # self.fusion=DualCrossModalAttention(in_dim,size=size)

        # self.cross_conv = nn.Conv2d(64 * 2, 64, 1, padding=0)
        self.trans=ConvBNReLU(2*in_dim,in_dim,3,1,1)
        self.fusion2=CIM(64,64)
    def forward(self, s, m,l, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""

        fus1 = self.trans(torch.cat([s,m], dim=1))
        out=self.fusion2(fus1,l)
        # attn_s, attn_m = torch.softmax(attn, dim=1).chunk(2, dim=1)
        # sm = attn_m * m + attn_s * s
        #
        # sm,l=self.fusion(sm,l)
        # out=self.cross_conv(torch.cat([sm,l],dim=1))


        # attn = self.trans(torch.cat([sl,sm], dim=1))
        # attn_sl, attn_sm = torch.softmax(attn, dim=1).chunk(2, dim=1)
        # lms = attn_sl * sl + attn_sm * sm

        # if return_feats:
            # return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return out