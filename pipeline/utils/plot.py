from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_evaluations(k_eval_df, save_dir, run):
    """
    dir_type: Directory type, which refers to if folders structured k/plots/eval or plots/eval/plottype/run.png
    """

    plt.figure(figsize=(10, 7))

    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(k_eval_df['Epoch'], k_eval_df['Train Accuracy'], label='Training')
    plt.plot(k_eval_df['Epoch'], k_eval_df['Validation Accuracy'], label='Validation')
    plt.legend()
    name = str(run)+'_accuracy.png'
    plt.savefig(save_dir+'/plots/accuracies/'+name)
    plt.clf()

    # Plot train/validation auc
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.plot(k_eval_df['Epoch'], k_eval_df['Train AUC'], label='Training AUC')
    plt.plot(k_eval_df['Epoch'], k_eval_df['Validation AUC'], label='Validation AUC')
    plt.legend()
    name = 'AUC/'+str(run)+'_AUC.png'
    plt.savefig(save_dir+'/plots'+'/'+name)
    plt.clf()

    # Plot training loss
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(k_eval_df['Epoch'], k_eval_df['Train Loss'], label='Training')
    plt.plot(k_eval_df['Epoch'], k_eval_df['Validation Loss'], label='Validation')
    plt.legend()
    name = str(run)+'_loss.png'
    plt.savefig(save_dir+'/plots/loss/'+name)
    plt.clf()

    # Plot points in margin
    plt.title('Points in Margin')
    plt.xlabel('Epoch')
    plt.ylabel('Points in Margin')
    plt.plot(k_eval_df['Epoch'], k_eval_df['Train in Margin'], label='Train')
    plt.plot(k_eval_df['Epoch'], k_eval_df['Validation in Margin'], label='Validation')
    plt.legend()
    name = 'margin/'+str(run)+'_points_in_margin.png'
    plt.savefig(save_dir+'/plots'+'/'+name)
    plt.clf()


def plot_params(param_dict, save_dir, run):
    os.makedirs(save_dir+'/plots/parameters/'+str(run))
    plt.figure(figsize=(12,8))
    for name, params in param_dict.items():
        x = range(params.shape[1])
        for param_series in params:
            plt.plot(x, param_series)
        plt.title(name)
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        
        plt.savefig(save_dir+'/plots/parameters/'+str(run)+'/'+name+'.png', dpi=150)
        plt.clf()


def plot_confusion(train_confusion, valid_confusion, save_dir, run):
    try:
        plt.figure(figsize=(9,6))
        total_train = np.sum(train_confusion[0])
        total_valid = np.sum(valid_confusion[0])

        plt.title('Train Confusion Matrix')
        plt.xlabel('Epoch')
        plt.ylabel('Percent Classified')
        plt.plot(range(len(train_confusion)), train_confusion[:,0,0]/total_train, label='True Negative')
        plt.plot(range(len(train_confusion)), train_confusion[:,0,1]/total_train, label='False Positive')
        plt.plot(range(len(train_confusion)), train_confusion[:,1,1]/total_train, label='True Positive')
        plt.plot(range(len(train_confusion)), train_confusion[:,1,0]/total_train, label='False Negative')
        plt.legend()
        plt.savefig(save_dir+'/plots/confusion/'+str(run)+'/train_confusion_'+str(run)+'.png')
        plt.clf()

        plt.title('Validation Confusion Matrix')
        plt.xlabel('Epoch')
        plt.ylabel('Percent Classified')
        plt.plot(range(len(valid_confusion)), valid_confusion[:,0,0]/total_valid, label='True Negative')
        plt.plot(range(len(valid_confusion)), valid_confusion[:,0,1]/total_valid, label='False Positive')
        plt.plot(range(len(valid_confusion)), valid_confusion[:,1,1]/total_valid, label='True Positive')
        plt.plot(range(len(valid_confusion)), valid_confusion[:,1,0]/total_valid, label='False Negative')
        plt.legend()
        plt.savefig(save_dir+'/plots/confusion/'+str(run)+'/valid_confusion_'+str(run)+'.png')
        plt.clf()

        plt.title('True Positive')
        plt.xlabel('Epoch')
        plt.ylabel('Percent Classified')
        plt.plot(range(len(train_confusion)), train_confusion[:,1,1]/total_train, label='Train')
        plt.plot(range(len(valid_confusion)), valid_confusion[:,1,1]/total_valid, label='Validation')
        plt.legend()
        plt.savefig(save_dir+'/plots/confusion/'+str(run)+'/true_positive_'+str(run)+'.png')
        plt.clf()

        plt.title('False Negative')
        plt.xlabel('Epoch')
        plt.ylabel('Percent Classified')
        plt.plot(range(len(train_confusion)), train_confusion[:,1,0]/total_train, label='Train')
        plt.plot(range(len(valid_confusion)), valid_confusion[:,1,0]/total_valid, label='Validation')
        plt.legend()
        plt.savefig(save_dir+'/plots/confusion/'+str(run)+'/false_negative_'+str(run)+'.png')
        plt.clf()

        plt.title('True Negative')
        plt.xlabel('Epoch')
        plt.ylabel('Percent Classified')
        plt.plot(range(len(train_confusion)), train_confusion[:,0,0]/total_train, label='Train')
        plt.plot(range(len(valid_confusion)), valid_confusion[:,0,0]/total_valid, label='Validation')
        plt.legend()
        plt.savefig(save_dir+'/plots/confusion/'+str(run)+'/true_negative_'+str(run)+'.png')
        plt.clf()

        plt.title('False Positive')
        plt.xlabel('Epoch')
        plt.ylabel('Percent Classified')
        plt.plot(range(len(train_confusion)), train_confusion[:,0,1]/total_train, label='Train')
        plt.plot(range(len(valid_confusion)), valid_confusion[:,0,1]/total_valid, label='Validation')
        plt.legend()
        plt.savefig(save_dir+'/plots/confusion/'+str(run)+'/false_positive_'+str(run)+'.png')
        plt.clf()

        plt.title('Total Positive')
        plt.xlabel('Epoch')
        plt.ylabel('Percent Classified Positive')
        plt.plot(range(len(train_confusion)), (train_confusion[:,0,1]+train_confusion[:,1,1])/total_train, label='Train')
        plt.plot(range(len(valid_confusion)), (valid_confusion[:,0,1]+valid_confusion[:,1,1])/total_valid, label='Validation')
        plt.legend()
        plt.savefig(save_dir+'/plots/confusion/'+str(run)+'/total_positive_'+str(run)+'.png')
        plt.clf()

        plt.title('Total Negative')
        plt.xlabel('Epoch')
        plt.ylabel('Percent Classified Negative')
        plt.plot(range(len(train_confusion)), (train_confusion[:,0,0]+train_confusion[:,1,0])/total_train, label='Train')
        plt.plot(range(len(valid_confusion)), (valid_confusion[:,0,0]+valid_confusion[:,1,0])/total_valid, label='Validation')
        plt.legend()
        plt.savefig(save_dir+'/plots/confusion/'+str(run)+'/total_negative_'+str(run)+'.png')
        plt.clf()
    except:
        print("Confusion matrices too small again?")


def plot_classifications(train_pred, valid_pred, train_labels, valid_labels, test_correct, save_dir, run):
    sns.heatmap(train_pred)
    plt.title('Training Set Correct')
    plt.savefig(save_dir+'/classifications/'+str(run)+'_train_correct.png')
    plt.clf()
    sns.heatmap(valid_pred)
    plt.title('Validation Set Correct')
    plt.savefig(save_dir+'/classifications/'+str(run)+'_validation_correct.png')
    plt.clf()
    sns.heatmap(test_correct)
    plt.title('Test Set Correct Correct')
    plt.savefig(save_dir+'/classifications/'+str(run)+'_test_correct.png')
    plt.clf()

    train_pos_i = np.where(train_labels == 1)
    train_neg_i = np.where(train_labels == -1)
    train_correct_only_pos = train_pred[train_pos_i]
    train_correct_only_neg = train_pred[train_neg_i]
    sns.heatmap(train_correct_only_pos)
    plt.title('Training Set Correct Positive')
    plt.savefig(save_dir+'/classifications/only_pos/'+str(run)+'_train_correct.png')
    plt.clf()
    sns.heatmap(train_correct_only_neg)
    plt.title('Training Set Correct Negative')
    plt.savefig(save_dir+'/classifications/only_neg/'+str(run)+'_train_correct.png')
    plt.clf()

    valid_pos_i = np.where(valid_labels == 1)
    valid_neg_i = np.where(valid_labels == -1)
    valid_correct_only_pos = valid_pred[valid_pos_i]
    valid_correct_only_neg = valid_pred[valid_neg_i]
    sns.heatmap(valid_correct_only_pos)
    plt.title('Validation Set Correct Positive')
    plt.savefig(save_dir+'/classifications/only_pos/'+str(run)+'_valid_correct.png')
    plt.clf()
    sns.heatmap(valid_correct_only_neg)
    plt.title('Validation Set Correct Negative')
    plt.savefig(save_dir+'/classifications/only_neg/'+str(run)+'_valid_correct.png')
    plt.clf()