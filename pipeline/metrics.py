import torch
from sklearn import metrics


def euclidean_distance(g_1, g_2):
    """Returns Euclidean Distance for batch of graph pairs."""
    distance = torch.sum((g_1 - g_2) ** 2, axis=-1)
    return distance


def embedding_accuracy(g_1, g_2, labels, margin):
    """
    Returns accuracy for a batches of graph embeddings using given labels"""

    batch_size = g_1.shape[0]
    distance = None

    distance = euclidean_distance(g_1, g_2)
    correct = torch.sign(torch.relu(labels * (1 - distance)))
    not_in_margin = torch.sum(torch.sign(torch.abs(1 - distance) - margin))
    accuracy = torch.sum(correct) / batch_size
    in_margin = (batch_size - not_in_margin) / batch_size

    return accuracy, distance, correct, in_margin


def similarity_accuracy(similarities, labels, margin):
    """Returns accuracy for a batch of similarity scores and labels.
    
    Returns:
        accuracy: accuracy of this batch, excluding predictions in margin
        correct: binary tensor representing batch's correct predictions
        in_margin: ratio of points in margin for this batch
    """

    batch_size = similarities.shape[0]
    correct = torch.sign(torch.relu(labels * similarities.squeeze()))
    accuracy = torch.sum(correct) / batch_size
    not_in_margin = torch.sum(torch.sign(torch.relu(torch.abs(similarities.squeeze()) - margin)))
    in_margin = (batch_size - not_in_margin) / batch_size

    return accuracy, correct, in_margin


def embedding_auc(g_1, g_2, labels):
    """Returns AUC of the ROC curve for the given batch of embeddings and labels."""

    auc = None
    
    # Cutoff negative values at -1 for prediction confidence
    distance = torch.clamp(1 - euclidean_distance(g_1, g_2), min=-1.0)
    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), distance.cpu().detach().numpy())
    auc = metrics.auc(fpr, tpr)
    return auc


def similarity_auc(similarities, labels):
    """Returns AUC of the ROC curve for the given batch of similarities and labels."""
        
    # Similarities centered at zero
    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu().detach().numpy(), similarities.cpu().detach().numpy())
    auc = metrics.auc(fpr, tpr)

    return auc