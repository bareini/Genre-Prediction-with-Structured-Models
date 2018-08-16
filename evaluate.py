import numpy as np
class Evaluate:
    """
    this class evaluates the results and creates the confusion matrix
    """

    def __init__(self,model):
        self.accuracy = []
        self.recall = []
        self.precision = []
        self.model = model

        self.acc_per_dec = []
        self.recall_per_dev = []
        self.precision_per_dev = []

    def evaluate_per_dev(self):
        """

        :return: 3 lists : acc_per_dec, recall_per_dev, precision_per_dev
        """
        i = 0
        for key in self.model.dict_notes_per_device.keys():
            end_list = i + len(self.model.dict_notes_per_device[key])
            self.acc_per_dec.append(np.mean(self.accuracy[i:end_list]))
            self.recall_per_dev.append(np.mean(self.recall[i:end_list]))
            self.precision_per_dev.append(np.mean(self.precision[i:end_list]))
            i = end_list



    def calc_acc_recall_precision(self, pred_labels):
        """
        This function calculate the accuracy, recall and precision
        :param pred_labels prediction labels, true_labels
        :return accuracy, recall, precision
        """
        true_labels = self.model.true_genres
        for i in range(len(true_labels)):
            j = 0
            correct_for_label = 0
            true_i = true_labels[i].split(',')
            pred_i = pred_labels[i].split(',')
            for j in range(len(pred_i)):
                pred = pred_i[j]
                if pred in true_i:
                    correct_for_label += 1
            self.accuracy.append(correct_for_label/len(list(set(true_i+pred_i))))
            self.recall.append(correct_for_label/len(true_i))
            self.precision.append(correct_for_label/len(pred_i))

        avg_accuracy = np.mean(self.accuracy)
        avg_recall = np.mean(self.recall)
        avg_precision = np.mean(self.precision)


        return avg_accuracy, avg_recall, avg_precision

    def bin_acc_recall_precision(self, pred_labels):
        bin_acc = []
        true_labels = self.model.true_genres
        for i in range(len(true_labels)):
            bin_acc.append(1 if true_labels[i] == pred_labels[i] else 0)

        avg_bin_acc = np.mean(bin_acc)

        return avg_bin_acc
