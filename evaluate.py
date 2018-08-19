import numpy as np
class Evaluate:
    """
    this class evaluates the results and creates the confusion matrix
    """

    def __init__(self,model):
        self.accuracy = []
        self.recall = []
        self.precision = []
        self.f1 = []
        self.model = model

        self.acc_per_dec = []
        self.recall_per_dev = []
        self.precision_per_dev = []
        self.f1_per_dev = []

    def evaluate_per_dev(self):
        """

        :return: 3 lists : acc_per_dec, recall_per_dev, precision_per_dev
        """
        i = 0
        for key in self.model.dict_notes_per_device.keys():
            end_list = i + len(self.model.dict_notes_per_device[key])
            self.acc_per_dec.append(np.mean(self.accuracy[i:end_list]))
            recall_i = np.mean(self.recall[i:end_list])
            self.recall_per_dev.append(recall_i)
            precision_i = np.mean(self.precision[i:end_list])
            self.precision_per_dev.append(precision_i)
            self.f1_per_dev.append((2 * precision_i * recall_i) / (precision_i + recall_i))
            i = end_list



    def calc_acc_recall_precision(self, pred_labels):
        """
        This function calculate the accuracy, recall and precision
        :param pred_labels prediction labels, true_labels
        :return accuracy, recall, precision
        """
        true_labels = self.model.true_genres
        avg_accuracy = 0.0
        avg_recall = 0.0
        avg_precision = 0.0
        for i in range(len(true_labels)):
            j = 0
            accuracy_i = 0.0
            recall_i = 0.0
            correct_for_label = 0
            true_i = true_labels[i].split(',')
            pred_i = pred_labels[i].split(',')
            for j in range(len(pred_i)):
                pred = pred_i[j]
                if pred in true_i:
                    correct_for_label += 1
            accuracy_i = correct_for_label/len(list(set(true_i+pred_i)))
            self.accuracy.append(accuracy_i)
            recall_i = correct_for_label/len(true_i)
            self.recall.append(recall_i)
            precision_i = correct_for_label/len(pred_i)
            self.precision.append(precision_i)

            f1_i = (((2 * precision_i * recall_i) / (precision_i + recall_i)) if precision_i + recall_i != 0.0 else 0.0)

            self.f1.append(f1_i)

        avg_accuracy = np.mean(self.accuracy)
        avg_recall = np.mean(self.recall)
        avg_precision = np.mean(self.precision)

        f1_of_avg = ((2 * avg_precision * avg_recall) / (avg_precision + avg_recall)) if avg_precision + avg_recall != 0.0 else 0.0


        return avg_accuracy, avg_recall, avg_precision, f1_of_avg

    def bin_acc_recall_precision(self, pred_labels):
        bin_acc = []
        true_labels = self.model.true_genres
        for i in range(len(true_labels)):
            bin_acc.append(1 if true_labels[i] == pred_labels[i] else 0)

        avg_bin_acc = np.mean(bin_acc)

        return avg_bin_acc
