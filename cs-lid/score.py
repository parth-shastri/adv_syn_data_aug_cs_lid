from tensorflow.keras.models import Model, load_model
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import seaborn as sns
from dataloader import test_data, test_labels
from sklearn.metrics import classification_report, r2_score
from config import CONFIG


def compare_models(test_data, test_labels, **kwargs):
    arg_list = []
    names = []
    labels = np.array(CONFIG["label_names"])

    y_true = test_labels

    for name, arg in kwargs.items():
        arg_list.append(arg)
        names.append(" ".join(name.split("_")))

    n = len(arg_list)
    lims = [i for i in range(n + 1)]

    reports = []
    f1 = []
    recalls = []
    uar = []
    class_wise_rec = []
    class_wise_prec = []
    for i, model in enumerate(arg_list):
        preds = model.predict(test_data, verbose=1)
        y_pred = np.argmax(preds, axis=-1)
        report = classification_report(y_true, y_pred, output_dict=True)
        # print(report)
        f1score = report["2"]["f1-score"]
        recall = report["2"]["recall"]
        classwise_rec = []
        classwise_prec = []
        for keys, values in report.items():
            try:
                rec = report[keys]["recall"]
                prec = report[keys]["precision"]
                classwise_rec.append(rec)
                classwise_prec.append(prec)
            except TypeError:
                print("..pass")

        uar.append(sum(classwise_rec) / len(classwise_rec))
        class_wise_rec.append(classwise_rec)
        class_wise_prec.append(classwise_prec)
        reports.append(report)
        f1.append(f1score)
        recalls.append(recall)
        print("{}".format(names[i]))
        print("UAR-{}".format(sum(classwise_rec) / len(classwise_rec)))
        print(classification_report(y_true, y_pred, digits=4))

    print(plt.style.available)
    print(max(uar))
    plt.style.use("seaborn")
    fig = plt.figure()
    plt.barh(y=lims, width=[0.0] + uar, height=0.5,
             color=["cornflowerblue"] + ["royalblue" if ua == max(uar) else "cornflowerblue" for ua in uar])
    plt.ylim((0, n + 1))
    plt.yticks(lims, [""] + names, rotation=45, fontsize="large")
    # plt.ylabel("Runs")
    plt.xlabel("UAR for ""Hindi-English""", fontweight="bold")
    fig.suptitle("UAR vs models", fontweight="bold", fontsize="x-large")
    fig.savefig(r"C:\Users\shast\OneDrive\Desktop\imgs\UAR_comparison.png")
    plt.show()
    return class_wise_rec, class_wise_prec


def plot_prec_recall(recalls, precs):
    barWidth = 0.9
    r1 = [1, 7, 13]
    r2 = [2, 8, 14]
    r3 = [3, 9, 15]
    r4 = [4, 10, 16]
    r5 = [5, 11, 17]
    fig = plt.figure()
    plt.bar(r1, recalls[0][:3], width=barWidth, color=(0.3, 0.1, 0.4, 0.8), label='baseline')
    plt.bar(r2, recalls[1][:3], width=barWidth, color=(0.3, 0.3, 0.4, 0.8), label='SpecAugment')
    plt.bar(r3, recalls[2][:3], width=barWidth, color=(0.3, 0.5, 0.4, 0.8), label='Time stretch')
    plt.bar(r4, recalls[3][:3], width=barWidth, color=(0.3, 0.7, 0.4, 0.8), label='Pitch shift')
    plt.bar(r5, recalls[4][:3], width=barWidth, color=(0.3, 0.9, 0.4, 0.8), label='proposed')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks([3, 9, 15], ["English", "Hindi", "Hindi-English"])
    plt.ylabel("Recall value", fontweight="bold", fontsize="large")
    fig.suptitle("Class-wise recall trend", fontweight="bold", fontsize="x-large")
    fig.savefig(r"C:\Users\shast\OneDrive\Desktop\imgs\classwise_recall.png", bbox_inches="tight")

    fig = plt.figure()
    plt.bar(r1, precs[0][:3], width=barWidth, color=(0.3, 0.1, 0.2, 0.8), label='baseline')
    plt.bar(r2, precs[1][:3], width=barWidth, color=(0.3, 0.3, 0.2, 0.8), label='SpecAugment')
    plt.bar(r3, precs[2][:3], width=barWidth, color=(0.3, 0.5, 0.2, 0.8), label='Time stretch')
    plt.bar(r4, precs[3][:3], width=barWidth, color=(0.3, 0.7, 0.2, 0.8), label='Pitch shift')
    plt.bar(r5, precs[4][:3], width=barWidth, color=(0.3, 0.9, 0.2, 0.8), label='proposed')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks([3, 9, 15], ["English", "Hindi", "Hindi-English"])
    plt.ylabel("Precision value", fontweight="bold", fontsize="large")
    fig.suptitle("Class-wise precision trend", fontweight="bold", fontsize="x-large")
    fig.savefig(r"C:\Users\shast\OneDrive\Desktop\imgs\classwise_precision.png", bbox_inches="tight")
    fig.show()


def plot_confusion_matrix(test_data, test_labels, model):
    labels = np.array(CONFIG["label_names"])
    preds = model.predict(test_data, verbose=1)
    y_pred = np.argmax(preds, axis=-1)
    print(y_pred)
    y_true = test_labels
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=CONFIG["label_names"],
                yticklabels=CONFIG["label_names"],
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


if __name__ == "__main__":
    gan_model_path = "models_/gan_synthetic_data_model_(1).h5"
    gan_model = tf.keras.models.load_model(gan_model_path)
    print("loaded the model from {}".format(gan_model_path))

    no_aug_model_path = "models_/imbalanced.h5"
    no_aug_model = tf.keras.models.load_model(no_aug_model_path)
    print("loaded the model from {}".format(no_aug_model_path))

    spec_aug_model_path = "models_/spec_aug.h5"
    spec_aug_model = tf.keras.models.load_model(spec_aug_model_path)
    print("loaded the model from {}".format(spec_aug_model_path))

    time_model_path = "models_/time_stretch.h5"
    time_aug_model = tf.keras.models.load_model(time_model_path)
    print("loaded the model from {}".format(time_model_path))

    pitch_model_path = "models_/pitch_shift.h5"
    pitch_aug_model = tf.keras.models.load_model(pitch_model_path)
    print("loaded the model from {}".format(pitch_model_path))

    test_labels = [label.numpy() for _, label in test_data]
    # print(test_labels)
    recalls, precs = compare_models(test_data.batch(CONFIG["batch_size"]), test_labels,
                                    baseline=no_aug_model,
                                    spec_augment=spec_aug_model,
                                    time_stretch=time_aug_model,
                                    pitch_shift=pitch_aug_model,
                                    proposed=gan_model)
    # plot_confusion_matrix(test_data.batch(CONFIG["batch_size"]), test_labels, gan_model)
    # plot_confusion_matrix(test_data.batch(CONFIG["batch_size"]), test_labels, no_aug_model)
    # plot_confusion_matrix(test_data.batch(CONFIG["batch_size"]), test_labels, spec_aug_model)
    # print(recalls, precs)
    plot_prec_recall(recalls, precs)
