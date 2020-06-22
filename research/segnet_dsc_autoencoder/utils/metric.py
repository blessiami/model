import torch


class Metric(object):
    def __init__(self, n_classes, device, ignore=255):
        super(Metric, self).__init__()
        self._n_classes = n_classes
        self._ignore = ignore

        self._n_pixels = torch.zeros(self._n_classes, dtype=torch.int64, requires_grad=False).to(device)
        self._n_intersections = torch.zeros(self._n_classes, dtype=torch.int64, requires_grad=False).to(device)
        self._n_unions = torch.zeros(self._n_classes, dtype=torch.int64, requires_grad=False).to(device)

    def reset(self):
        self._n_pixels.fill_(0)
        self._n_intersections.fill_(0)
        self._n_unions.fill_(0)

    def update(self, predictions, labels):
        with torch.no_grad():
            pass
            # mask = labels.ne(self._ignore).type(dtype=torch.uint8)
            #
            # for i in range(self._n_classes):
            #     prediction = predictions.eq(i) * mask
            #     label = labels.eq(i) * mask
            #
            #     n_prediction = prediction.sum()
            #     n_label = label.sum()
            #     n_intersection = (prediction * label).sum()
            #
            #     self._n_pixels[i] += n_label
            #     self._n_intersections[i] += n_intersection
            #     self._n_unions[i] += (n_prediction + n_label - n_intersection)

    def calculate(self, valid=False):
        with torch.no_grad():
            # epsilon = 1e-9
            #
            # n_pixels = self._n_pixels.type(dtype=torch.float64)
            # n_intersections = self._n_intersections.type(dtype=torch.float64)
            # n_unions = self._n_unions.type(dtype=torch.float64)
            #
            # pixel_accuracy = n_intersections.sum() / (n_pixels.sum() + epsilon)
            #
            # if valid:
            #     n_effective = (n_pixels != 0).sum().type(dtype=torch.float64)
            #     mean_accuracy = (n_intersections / (n_pixels + epsilon)).sum() / n_effective
            #     mean_iou = (n_intersections / (n_unions + epsilon)).sum() / n_effective
            # else:
            #     mean_accuracy = (n_intersections / (n_pixels + epsilon)).mean()
            #     mean_iou = (n_intersections / (n_unions + epsilon)).mean()
            #
            # frequency_weighted_iou = (n_pixels * n_intersections / (n_unions + epsilon)).sum() \
            #                          / (n_pixels.sum() + epsilon)
            #
            # iou = n_intersections / (n_unions + epsilon)
            # true_positive = n_intersections
            # false_positive = n_unions - n_pixels
            # false_negative = n_pixels - n_intersections
            #
            # precision = true_positive / (true_positive + false_positive + epsilon)
            # recall = true_positive / (true_positive + false_negative + epsilon)
            # f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
            pixel_accuracy = 0
            mean_accuracy = 0
            mean_iou = 0
            frequency_weighted_iou = 0
            iou = 0
            true_positive = 0
            false_positive = 0
            false_negative = 0
            precision = 0
            recall = 0
            f1_score = 0

            return {'pixel_accuracy': pixel_accuracy,
                    'mean_accuracy': mean_accuracy,
                    'mean_iou': mean_iou,
                    'frequency_weighted_iou': frequency_weighted_iou,
                    'iou': iou,
                    'true_positive': true_positive,
                    'false_positive': false_positive,
                    'false_negative': false_negative,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score}
