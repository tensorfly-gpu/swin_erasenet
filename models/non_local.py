import paddle
from paddle import nn


class NonLocalBlock(nn.Layer):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.conv_theta = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.conv_g = nn.Conv2D(channel, self.inter_channel, kernel_size=1, stride=1, bias_attr=False)
        self.softmax = nn.Softmax(axis=1)
        self.conv_mask = nn.Conv2D(self.inter_channel, channel, kernel_size=1, stride=1, bias_attr=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x_phi = self.conv_phi(x)
        x_phi = paddle.reshape(x_phi, (b, c, -1))
        x_theta = self.conv_theta(x)
        x_theta = paddle.transpose(paddle.reshape(x_theta, (b, c, -1)), (0, 2, 1))
        x_g = self.conv_g(x)
        x_g = paddle.transpose(paddle.reshape(x_g, (b, c, -1)), (0, 2, 1))
        mul_theta_phi = paddle.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = paddle.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = paddle.transpose(mul_theta_phi_g, (0, 2, 1))
        mul_theta_phi_g = paddle.reshape(mul_theta_phi_g, (b, self.inter_channel, h, w))
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out
