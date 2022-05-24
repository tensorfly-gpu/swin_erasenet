import paddle
from paddle import nn
import paddle.nn.functional as F


def bce_loss(inputs, target):
    inputs = F.sigmoid(inputs)

    inputs = inputs.reshape([inputs.shape[0], -1])
    target = target.reshape([target.shape[0], -1])

    inputs = inputs
    target = target

    bce = paddle.nn.BCELoss()

    return bce(inputs, target)


class LossWithGAN_STE(nn.Layer):
    def __init__(self):
        super(LossWithGAN_STE, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mask, output, mm, gt):
        holeLoss = self.l1(mask * output, mask * gt)
        validAreaLoss = self.l1((1 - mask) * output, (1 - mask) * gt)
        mask_loss = bce_loss(mm, mask) + self.l1(mm, mask)

        image_loss = self.l1(output, gt)
        GLoss = 0.5 * mask_loss + 0.5 * holeLoss + 0.5 * validAreaLoss + 1.5 * image_loss
        return GLoss.sum()


class LossWithSwin(nn.Layer):
    def __init__(self):
        super(LossWithSwin, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mask, output, mm, gt):
        holeLoss = self.l1(mask * output, mask * gt)
        validAreaLoss = self.l1((1 - mask) * output, (1 - mask) * gt)
        mask_loss = self.l1(mm, mask)

        image_loss = self.l1(output, gt)
        GLoss = 1.0 * mask_loss + 0.5 * holeLoss + 0.5 * validAreaLoss + 1.5 * image_loss
        return GLoss.sum()