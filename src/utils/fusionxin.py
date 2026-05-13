import torch
from torch import nn
#from tensorboardX import SummaryWriter
import src.utils.layers as _layers
import torch.nn.functional as F
import numpy as np
import torch.fft as afft
from torch.autograd import Variable
class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.
    Args:
        output_dim: output dimension for compact bilinear pooling.
        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.
        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.
        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.
        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.
        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, cuda=True,
                 rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim))

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim))

        if cuda:
            self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.cuda()
            self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.cuda()

    def forward(self, bottom1, bottom2):
        """
        bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
        """
        assert bottom1.size(1) == self.input_dim1 and \
            bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        fft1 = afft.fft(sketch_1)
        fft2 = afft.fft(sketch_2)

        fft_product = fft1 * fft2

        cbp_flat = afft.ifft(fft_product).real

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)
        else:
            cbp = cbp.permute(0, 3, 1, 2)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim ==
               1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()

class ASPP(nn.Module):
    def __init__(self, inc, outc, d=[1, 3, 5, 7]):
        super(ASPP, self).__init__()
        self.conv1 = _layers.conv3x3_bn_relu_d(inc, outc // 4, d[0])
        self.conv2 = _layers.conv3x3_bn_relu_d(inc, outc // 4, d[1])
        self.conv3 = _layers.conv3x3_bn_relu_d(inc, outc // 4, d[2])
        self.conv4 = _layers.conv3x3_bn_relu_d(inc, outc // 4, d[3])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat((x1, x2, x3, x4), 1)
        return out


class Biliner_Fusion12(nn.Module):
    def __init__(self, inc, outc, d=[1, 2, 3, 4]):
        super(Biliner_Fusion12, self).__init__()
        # print('using bilinear 12 model')
        self.xb = CompactBilinearPooling(inc, inc, outc, sum_pool=False)

        self.att_map = nn.Sequential(_layers.conv3x3xbnxrelu(outc + inc * 2, outc),
                                     _layers.conv3x3xbnxrelu(outc, outc),
                                     nn.Conv2d(outc, 2, kernel_size=3, stride=1, padding=1),
                                     nn.Sigmoid())

        self.ff = _layers.BasicBlock(inc * 2, outc)
        self.fusion = ASPP(outc * 2, outc, [1,2,3,4])

    def forward(self, x, y):
        bf = self.xb(x, y)
        bf = F.normalize(bf, dim=1)
        ########attention
        bfin = torch.cat((bf, x, y), 1)
        maps = self.att_map(bfin)
        fxy = torch.cat((x * maps[:, 0:1, :, :], y * maps[:, 1:2, :, :]), 1)
        ########attention
        ff = self.ff(fxy)
        ########
        fusion = torch.cat((ff, bf), 1)
        fusion = self.fusion(fusion)
        return fusion

    def forward_for_visual(self, x, y):
        bf = self.xb(x, y)
        bf = F.normalize(bf, dim=1)
        ########attention
        bfin = torch.cat((bf, x, y), 1)
        maps = self.att_map(bfin)
        fxy = torch.cat((x * maps[:, 0:1, :, :], y * maps[:, 1:2, :, :]), 1)
        ########attention
        ff = self.ff(fxy)
        ########
        fusion = torch.cat((ff, bf), 1)
        fusion = self.fusion(fusion)
        return fusion, bf


class Skip22(nn.Module):
    def __init__(self, inc, outc, fm=Biliner_Fusion12):
        super(Skip22, self).__init__()
        self.fuse1 = fm(inc[0], outc[0])
        self.fuse2 = fm(inc[1], outc[1])
        self.fuse3 = fm(inc[2], outc[2], [1, 3, 5, 7])
        self.fuse4 = fm(inc[3], outc[3], [1, 3, 5, 7])

    def forward(self, r1, r2, r3, r4, d1, d2, d3, d4):
        out4 = self.fuse4(r4, d4)
        out3 = self.fuse3(r3, d3)
        out2 = self.fuse2(r2, d2)
        out1 = self.fuse1(r1, d1)
        return out1, out2, out3, out4

    # def forward_for_visual(self, r1, r2, r3, r4, d1, d2, d3, d4):
    #     #        out4 = self.fuse4(r4, d4)
    #     #        out3 = self.fuse3(r3, d3)
    #     #        out1, bf = self.fuse2.forward_for_visual(r2, d2)
    #     out1, bf = self.fuse1.forward_for_visual(r1, d1)
    #     return out1, bf


# model=Skip22([96,192,384,768],[96,192,384,768]).cuda()
# a=torch.rand(2,96,128,128).cuda()
# b=torch.rand(2,192,64,64).cuda()
# c=torch.rand(2,384,32,32).cuda()
# d=torch.rand(2,768,16,16).cuda()
# out1,out2,out3,out4=model(a,b,c,d,a,b,c,d)
# print(out1.size())