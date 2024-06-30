import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def false_positive(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fp = np.sum(pred - pred * label) + smooth
    fpr = round(fp * 100 / (np.sum((1.0 - label)) + smooth), 3)
    return fpr


def false_negative(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fn = np.sum(label - pred * label) + smooth
    fnr = round(fn * 100 / (np.sum(label) + smooth), 3)
    return fnr


def true_negative(pred, label):
    # specificity or True negative
    sensitivity = round(100 - false_positive(pred, label), 3)
    return sensitivity


def true_positive(pred, label):
    # sensitivity or True Positive
    specificity = round(100 - false_negative(pred, label), 3)
    return specificity


def dice_score(pred, label):
    # label = np.array(label > 0.5, "float32")
    # pred = np.array(pred > 0.5, "float32")

    inter = np.sum(pred * label)
    norm = np.sum(pred * pred) + np.sum(label * label)

    return round(2 * inter / norm * 100, 3)


def compare_img(img_1, img_2, norm=False):

    if not norm:
        img_1 = np.clip(img_1, 0, 1)
        img_2 = np.clip(img_2, 0, 1)
    else:
        # img_1 = normalize(img_1)
        # img_2 = normalize(img_2)
        img_1 = (img_1 - np.min(img_1)) / (np.max(img_1) - np.min(img_1))
        img_2 = (img_2 - np.min(img_2)) / (np.max(img_2) - np.min(img_2))

    psnr = round(peak_signal_noise_ratio(img_1, img_2), 4)
    ssim = round(structural_similarity(img_1, img_2, data_range=1), 4)

    nmse = round(np.sqrt(np.mean(np.square(img_1 - img_2))), 4)
    # mse = np.sqrt(mse)

    nmae = round(np.mean(np.abs(img_1 - img_2)), 4)
    return psnr, ssim, nmse, nmae


def nrmse(img_1, img_2):
    nmse = np.sqrt(np.mean(np.square(img_1 - img_2)) / np.mean(np.square(img_2))) * 100
    # print(nmse)
    return round(nmse, 2)


def sensitivity_multiclass_batch(preds, labels, num_classes, smooth=1e-5):
    batch_size = preds.shape[0]
    preds = np.argmax(preds, axis=1).reshape(batch_size, -1)
    labels = labels.reshape(batch_size, -1)
    sensitivity_per_class = np.zeros(num_classes)

    for c in range(num_classes):
        tp = np.sum((preds == c) & (labels == c), axis=1)
        fn = np.sum((preds != c) & (labels == c), axis=1)

        sensitivity = (tp + smooth) / (tp + fn + smooth)
        sensitivity_per_class[c] = np.mean(sensitivity)

    return sensitivity_per_class


def specificity_multiclass_batch(preds, labels, num_classes, smooth=1e-5):
    batch_size = preds.shape[0]
    preds = np.argmax(preds, axis=1).reshape(batch_size, -1)
    labels = labels.reshape(batch_size, -1)
    specificity_per_class = np.zeros(num_classes)

    for c in range(num_classes):
        tn = np.sum((preds != c) & (labels != c), axis=1)
        fp = np.sum((preds == c) & (labels != c), axis=1)

        specificity = (tn + smooth) / (tn + fp + smooth)
        specificity_per_class[c] = np.mean(specificity)

    return specificity_per_class


def dice_multiclass_batch(preds, labels, num_classes, smooth=1e-5):
    batch_size = preds.shape[0]
    preds = np.argmax(preds, axis=1).reshape(batch_size, -1)
    labels = labels.reshape(batch_size, -1)
    dice_per_class = np.zeros(num_classes)

    for c in range(num_classes):
        pred_class = (preds == c).astype(np.float32)
        label_class = (labels == c).astype(np.float32)

        intersection = np.sum(pred_class * label_class, axis=1)
        dice = (2 * intersection + smooth) / (np.sum(pred_class, axis=1) + np.sum(label_class, axis=1) + smooth)
        dice_per_class[c] = np.mean(dice)

    return dice_per_class