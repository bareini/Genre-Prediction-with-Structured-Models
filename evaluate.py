class Evaluate:
    """
    this class evaluates the results and creates the confusion matrix
    """

    def __init__(self):
        pass

    def calc_acc_recall_precision(pred_labels, true_labels):
        """
        This function calculate the accuracy, recall and precision
        :param pred_labels prediction labels, true_labels
        :return accuracy, recall, precision
        """
        # todo one list, keep for every note
        accuracy = 0
        recall = 0
        precision = 0

        # pred_label_list = pred_labels.split(',')
        # true_label_list = true_labels.split(',')

        for i in range(len(true_labels)):
            j = 0
            correct_for_label = 0
            for j in range(len(pred_labels[i])):
                pred = pred_labels[i][j]
                if pred in true_labels[i]:
                    correct_for_label += 1
            accuracy += correct_for_label/len(list(set(true_labels[i]+pred_labels[i])))
            recall += correct_for_label/len(true_labels[i])
            precision += correct_for_label/len(pred_labels[i])

        accuracy = accuracy/len(true_labels)
        recall = recall/len(true_labels)
        precision = precision/len(true_labels)

        return accuracy, recall, precision