# # encoding: utf-8
# """
# @author:  liaoxingyu
# @contact: sherlockliao01@gmail.com
# """
# from bisect import bisect_right
# import torch


# # FIXME ideally this would be achieved with a CombinedLRScheduler,
# # separating MultiStepLR with WarmupLR
# # but the current LRScheduler design doesn't allow it

# class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(
#             self,
#             optimizer,
#             milestones,  # steps
#             gamma=0.1,
#             warmup_factor=1.0 / 3,
#             warmup_iters=500,
#             warmup_method="linear",
#             last_epoch=-1,
#     ):
#         if not list(milestones) == sorted(milestones):
#             raise ValueError(
#                 "Milestones should be a list of" " increasing integers. Got {}",
#                 milestones,
#             )

#         if warmup_method not in ("constant", "linear"):
#             raise ValueError(
#                 "Only 'constant' or 'linear' warmup_method accepted"
#                 "got {}".format(warmup_method)
#             )
#         self.milestones = milestones
#         self.gamma = gamma
#         self.warmup_factor = warmup_factor
#         self.warmup_iters = warmup_iters
#         self.warmup_method = warmup_method
#         super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         warmup_factor = 1
#         if self.last_epoch < self.warmup_iters:
#             if self.warmup_method == "constant":
#                 warmup_factor = self.warmup_factor
#             elif self.warmup_method == "linear":
#                 alpha = self.last_epoch / self.warmup_iters
#                 warmup_factor = self.warmup_factor * (1 - alpha) + alpha
#         return [
#             base_lr
#             * warmup_factor
#             * self.gamma ** bisect_right(self.milestones, self.last_epoch)
#             for base_lr in self.base_lrs
#         ]


# encoding: utf-8
from bisect import bisect_right
import math
import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class PromptSGImageEncoderLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Only controls param groups whose lr_role == 'image_encoder'.

    Epoch 1-5   : linearly increase 1e-6 -> 1e-5
    Epoch 6-10  : hold at 1e-5
    Epoch 11-end: cosine decay 1e-5 -> 5e-6

    Other param groups keep the lr assigned by optimizer.
    """

    def __init__(
        self,
        optimizer,
        image_encoder_base_lr=5e-6,
        image_encoder_init_lr=1e-6,
        image_encoder_peak_lr=1e-5,
        warmup_epochs=5,
        hold_epochs=10,
        total_epochs=60,
        last_epoch=-1,
    ):
        if hold_epochs < warmup_epochs:
            raise ValueError("hold_epochs must be >= warmup_epochs")
        if total_epochs <= hold_epochs:
            raise ValueError("total_epochs must be > hold_epochs")

        self.image_encoder_base_lr = image_encoder_base_lr
        self.image_encoder_init_lr = image_encoder_init_lr
        self.image_encoder_peak_lr = image_encoder_peak_lr
        self.warmup_epochs = warmup_epochs
        self.hold_epochs = hold_epochs
        self.total_epochs = total_epochs

        super(PromptSGImageEncoderLRScheduler, self).__init__(optimizer, last_epoch)

    def _get_image_encoder_lr(self, epoch_idx):
        # epoch_idx: 0-based because scheduler.step() is called once per epoch

        # epochs 1..5  -> idx 0..4
        if epoch_idx <= self.warmup_epochs - 1:
            if self.warmup_epochs == 1:
                return self.image_encoder_peak_lr
            alpha = epoch_idx / float(self.warmup_epochs - 1)
            return self.image_encoder_init_lr + alpha * (
                self.image_encoder_peak_lr - self.image_encoder_init_lr
            )

        # epochs 6..10 -> idx 5..9
        if epoch_idx <= self.hold_epochs - 1:
            return self.image_encoder_peak_lr

        # epochs 11..end -> cosine decay back to base lr
        decay_span = self.total_epochs - self.hold_epochs
        t = (epoch_idx - (self.hold_epochs - 1)) / float(decay_span)
        t = min(max(t, 0.0), 1.0)

        return self.image_encoder_base_lr + 0.5 * (
            self.image_encoder_peak_lr - self.image_encoder_base_lr
        ) * (1.0 + math.cos(math.pi * t))

    def get_lr(self):
        epoch_idx = self.last_epoch
        image_encoder_lr = self._get_image_encoder_lr(epoch_idx)

        lrs = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            if group.get("lr_role", "other") == "image_encoder":
                lrs.append(image_encoder_lr)
            else:
                lrs.append(base_lr)
        return lrs