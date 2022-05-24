# Program to evaluate the patches

# Written by Marley Sudbury (1838838)
# for CM3203 One Semester Individual Project

# Confidences are encoded as R G B
# (100, 0, 0) = 100% positive
# (0, 100, 0) = 100% negative
# (50, 50, 0) = 50% positive, 50% negative
# (0, 0, 0) = background

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

list_of_filenames = [
    "tumor_001",
    "tumor_002",
    "tumor_003",
    "tumor_004",
    "tumor_005",
    "tumor_006",
    "tumor_007",
    "tumor_008",
    "tumor_009",
    "tumor_010"
]

# list_of_filenames = [
#     "22070",
#     "22071",
#     "22081",
#     "22082",
#     "22083",
#     # "22111",
#     "22112",
#     "22113",
#     "22114",
#     "22158"
# ]

model = "alpha_p_n"
# normal = "no normal"
normal = "normal"

tp = 0
fp = 0
tn = 0
fn = 0
for filename in list_of_filenames:
    # Load the generated mask
    mask_path = "E:\\fyp\\predictions\\{}\\{}\\{}.png".format(
        model, normal, filename)

    # Load the ground truth mask
    # truth_path = "E:\\fyp\\Training Data !\\Head_Neck_Annotations\\PNG\\{}.png".format(
    #     filename)
    truth_path = "E:\\fyp\\Training Data !\\Cam16Annotations\\PNG\\{}.png".format(
        filename)

    generated = Image.open(mask_path)
    generated_pixel_map = generated.load()
    truth = Image.open(truth_path)
    truth_pixel_map = truth.load()

    # Make sure that the provided images are the same size
    if generated.width != truth.width or generated.height != truth.height:
        print("Mask mismatch!")
        exit()

    positive_missed_by_RGB_threshold = 0
    confidences = []
    truth = []

    for x in range(generated.width):
        for y in range(generated.height):
            g_pixel = generated_pixel_map[x, y]
            p_pixel = truth_pixel_map[x, y]
            if g_pixel[1] >= 50:
                # Classified as positive
                confidences.append(0.5 + ((g_pixel[1] / 2)) / 200)
                if p_pixel == (0, 100, 0):
                    tp += 1
                    truth.append(1)
                else:
                    fp += 1
                    truth.append(0)
            elif g_pixel[0] >= 50:
                confidences.append(((g_pixel[0] / 2)) / 200)
                # Classified as negative
                if p_pixel == (100, 0, 0):
                    tn += 1
                    truth.append(0)
                else:
                    truth.append(1)
                    fn += 1
            elif g_pixel == (0, 0, 0):
                # Ignored as background
                if p_pixel == (0, 100, 0):
                    positive_missed_by_RGB_threshold += 1
    # print(tp, fp, tn, fn)
    if positive_missed_by_RGB_threshold > 0:
        print("{} positive patches were missed due to the RGB threshold or normalisation error!!!!".format(
            positive_missed_by_RGB_threshold))

m = tf.keras.metrics.AUC(num_thresholds=100)
m.update_state(truth, confidences)
print("AUC:", m.result().numpy())
print("Accuracy:", (tp + tn) / (tp + fp + tn + fn))
print("Specificity:", tn / (tn + fp))
print("Sensitivity:", tp / (tp + fn))

# Adapated from https://androidkt.com/get-the-roc-curve-and-auc-for-keras-model/


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.axis('square')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


# False positive rate (1-true negative rate) of alpha_a_n
fpr = []
tpr = []
step = 0.01
for i in np.arange(0.0, 1 + step, step):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for j in range(len(confidences)):
        if confidences[j] < i:
            if truth[j] == 0:
                tn += 1
            else:
                fn += 1
        else:
            if truth[j] == 1:
                tp += 1
            else:
                fp += 1
    tpr.append(tp / (tp + fn))
    fpr.append(1 - (tn / (tn + fp)))

fpr[100] = 0
tpr[100] = 0
plot_roc_curve(fpr, tpr)
