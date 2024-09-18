import numpy as np
import torch


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Specificity(self):
        SP = self.confusion_matrix[0, 0] / (self.confusion_matrix[0, 0] + self.confusion_matrix[0, 1] + 1e-10)
        return SP

    def Sensitivity(self):
        SE = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0] + 1e-10)
        return SE

    def Dice(self):
        Dice = 2 * self.confusion_matrix[1, 1] / (2 * self.confusion_matrix[1, 1] + self.confusion_matrix[0, 1] + self.confusion_matrix[1, 0] + 1e-10)
        return Dice

    def Pixel_Accuracy(self):
        Acc = (self.confusion_matrix[1, 1] + self.confusion_matrix[0, 0]) / self.confusion_matrix.sum()
        return Acc

    def Mean_Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        MPA = np.nanmean(Acc)
        return MPA

    def Pixel_F1(self):
        precision = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[0, 1] + 1e-10)
        recall = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0] + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return f1

    def Intersection_over_Union(self):
        IoU = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[0, 1] + 1e-10)
        return IoU

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class) 
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




