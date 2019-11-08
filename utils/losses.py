import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class SegmentationFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(SegmentationFocalLoss, self).__init__()
        self.loss = FocalLoss(gamma=gamma)

    def forward(self, input, target):
        input = input['logits']
        return self.loss(input, target)


class RegressionFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(RegressionFocalLoss, self).__init__()
        self.loss = FocalLoss(gamma=gamma)

    def forward(self, input, target):
        input = input['logits_reg']
        image = target['image']
        image = torch.sum(image, dim=1, keepdim=True)
        target_class = target['mask'].unsqueeze(dim=1)
        target_wlh = target['wlh']
        target_h = target_wlh[:, 2, :, :].unsqueeze(dim=1)

        # print('input', input.size())
        # print('image', image.size())
        # print('target_class', target_class.size())
        # print('target_h', target_h.size())

        target_mask = target_class > 0
        image_mask = image > 0

        mask = target_mask * image_mask
        mask_expanded = mask.expand((
            input.size(0), input.size(1), input.size(2), input.size(3)
        ))

        # print('mask', mask.size())
        # print('mask nonzero', torch.sum(mask))
        # print('mask_expanded', mask_expanded.size())
        # print('mask_expanded nonzero', torch.sum(mask_expanded))

        input_masked = input[mask_expanded].view(1, input.size(1), -1)
        target_h_masked = target_h[mask].view(1, -1)

        # print('input_masked', input_masked.size())
        # print('target_h_masked', target_h_masked.size())
        # print('target', target.keys())

        return self.loss(input_masked, target_h_masked)


class RegressionLoss(nn.Module):
    def __init__(self):
        super(RegressionLoss, self).__init__()
        self.loss = nn.modules.loss.SmoothL1Loss()

    def forward(self, input, target):
        input = input['logits_reg']
        image = target['image']
        image = torch.sum(image, dim=1, keepdim=True)
        target_class = target['mask'].unsqueeze(dim=1)
        target_wlh = target['wlh']
        target_h = target_wlh[:, 2, :, :].unsqueeze(dim=1)

        # print('input', input.size())
        # print('image', image.size())
        # print('target_class', target_class.size())
        # print('target_h', target_h.size())

        target_mask = target_class > 0
        image_mask = image > 0

        mask = target_mask * image_mask
        mask_expanded = mask.expand((
            input.size(0), input.size(1), input.size(2), input.size(3)
        ))

        # print('mask', mask.size())
        # print('mask nonzero', torch.sum(mask))
        # print('mask_expanded', mask_expanded.size())
        # print('mask_expanded nonzero', torch.sum(mask_expanded))

        input_masked = input[mask_expanded].view(1, -1)
        target_h_masked = target_h[mask].view(1, -1)

        # print('input_masked', input_masked.size())
        # print('target_h_masked', target_h_masked.size())
        # print('target', target.keys())

        return self.loss(input_masked, target_h_masked)



def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        inputs = F.softmax(inputs, dim=1)
        inputs, targets = self.prob_flatten(inputs, targets)
        # print(inputs.shape, targets.shape)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, true, eps=1e-7):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)


def smooth_l1_loss(input, target, sigma=1.0, size_average=True):
    '''
    input: B, *
    target: B, *

    '''
    # smooth_l1_loss with sigma
    """
            (sigma * x)^2/2  if x<1/sigma^2
    f(x)=
            |x| - 1/(2*sigma^2) otherwise
    """
    assert input.shape == target.shape

    diff = torch.abs(input - target)

    mask = (diff < (1. / sigma ** 2)).detach().type_as(diff)

    output = mask * torch.pow(sigma * diff, 2) / 2.0 + (1 - mask) * (diff - 1.0 / (2.0 * sigma ** 2.0))
    loss = output.sum()
    if size_average:
        loss = loss / input.shape[0]

    return loss


class MultiLoss(nn.Module):
    def __init__(self, loss_funcs, coefs):
        super(MultiLoss, self).__init__()

        self.loss_funcs = nn.Sequential(*loss_funcs)
        self.coefs = coefs

    def forward(self, input, target):
        loss = 0

        for coef, loss_fun in zip(self.coefs, self.loss_funcs):
            cur_loss = coef * loss_fun(input, target)
            loss += cur_loss
        #                 print(cur_loss.item())

        #             print()

        return loss