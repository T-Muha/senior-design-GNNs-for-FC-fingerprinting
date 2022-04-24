import os
import json
import time
import numpy as np
import pandas as pd

from create_model import create_model
from utils.graph_utils import knn_graph, random_walks
from train import train
from evaluate import evaluate
from utils.results import make_directories, record_params
from utils.plot import plot_evaluations, plot_params, plot_confusion, plot_classifications
from FC_graphs import construct_dataset
from split_dataset import split, cross_validation_split

def run_pipeline(config, device=None):

    # Get some config values to save line space
    num_runs = config['num_runs']
    full_size = config['data']['full_size']
    batch_size = config['train']['batch_size']

    # Calculate set sizes
    if config['cross_validation']:
        test_size = int(full_size / num_runs)
        excl_test_size = full_size - test_size
        valid_size = int(excl_test_size * config['data']['valid_ratio'])
        train_size = excl_test_size - valid_size
    else:
        valid_size = int(full_size * config['data']['valid_ratio'])
        test_size = int(full_size * config['data']['test_ratio'])
        train_size = full_size - valid_size - test_size

    n_training_steps = int(train_size / batch_size)
    config['train']['n_training_steps'] = n_training_steps

    # Make the results directory structure
    save_name, save_dir = make_directories(config)

    # Save config values
    with open('results/'+save_name+'/config.json', 'w') as writer:
        json.dump(config, writer, indent=4)

    # Holds evaluation data, a list of dfs
    eval_dfs = []

    # Classification data, lists of dfs
    train_class_dfs = []
    valid_class_dfs = []
    test_class_dfs = []

    # Final train/test accuracy for each run
    test_accs = []
    test_aucs = []

    ids_dict = {}

    if config['cross_validation']:
        full_dataset, full_labels, ids, construct_dataset(config)

    start_time = time.time()

    for run in range(num_runs):
        print("Run %i" % (run+1))
        run_start = time.time()
        
        # Build and/or split datasets
        if config['cross_validation']:
            (train_pairs, train_labels, train_ids,
            valid_pairs, valid_labels, valid_ids,
            test_pairs, test_labels, test_ids) = cross_validation_split(full_dataset, full_labels, ids, valid_size, test_size, run)
        else:
            full_dataset, full_labels, ids = construct_dataset(config)
            (train_pairs, train_labels, train_ids,
            valid_pairs, valid_labels, valid_ids,
            test_pairs, test_labels, test_ids) = split(full_dataset, full_labels, ids, train_size, valid_size, test_size)

        # Structures to hold metrics
        param_dict = {}
        train_confusion = []
        valid_confusion = []

        # This run's evaluation data
        run_eval = []

        # Obtain binary mean knn graph from training data
        k_edges = config['model']['knn_edges']
        knn_G = knn_graph(train_pairs, k_edges)
        if config['model']['type'] == 'hs_gcn':
            num_walks = config['model']['random_walk']['num_walks']
            walk_len = config['model']['random_walk']['walk_len']
            window_size = config['model']['random_walk']['window_size']
            knn_G = random_walks(knn_G, num_walks, walk_len, window_size)

        model, optimizer = create_model(config, knn_G)
        model.to(device)

        decreased_lr = False

        for epoch in range(config['train']['epochs']):
            eval_row = {'Epoch': epoch}

            if epoch % 100 == 0:
                print("Epoch %i" % epoch)

            grad_norm = train(device, model, optimizer, train_pairs, train_labels, config)
            train_loss, train_acc, train_auc, train_dist, train_correct, train_in_margin, train_conf_matrix = evaluate(device, model, train_pairs, train_labels, config, shuffle=True)

            print("Train Acc: %0.4f   Train Loss: %0.4f   Grad Norm: %0.2f" % (train_acc, train_loss, grad_norm), end='    ')
            eval_row['Train Accuracy'] = train_acc
            eval_row['Train Loss'] = train_loss
            eval_row['Train AUC'] = train_auc
            eval_row['Train in Margin'] = train_in_margin
            eval_row['Gradient Norm'] = grad_norm
            train_confusion.append(train_conf_matrix)

            valid_loss, valid_acc, valid_auc, valid_dist, valid_correct, valid_in_margin, valid_conf_matrix = evaluate(device, model, valid_pairs, valid_labels, config)

            print("Valid Loss: %0.4f   Valid Acc: %0.4f" % (valid_loss, valid_acc))
            eval_row['Validation Loss'] = valid_loss
            eval_row['Validation Accuracy'] = valid_acc
            eval_row['Validation AUC'] = valid_auc
            eval_row['Validation in Margin'] = valid_in_margin
            valid_confusion.append(valid_conf_matrix)

            # Decrease lr if conditions met
            if config['train']['decrease_lr'] and not decreased_lr and train_loss < config['train']['decrease_lr_at']:
                decreased_lr = True
                for g in optimizer.param_groups:
                    g['lr'] = config['train']['secondary_lr']
                    print('decreased lr')

            # Record validation distances
            if epoch == 0:
                train_distances = train_dist
                valid_distances = valid_dist
                train_pred = train_correct
                valid_pred = valid_correct
            else:
                train_distances = np.append(train_distances, train_dist, axis=1)
                valid_distances = np.append(valid_distances, valid_dist, axis=1)
                train_pred = np.append(train_pred, train_correct, axis=1)
                valid_pred = np.append(valid_pred, valid_correct, axis=1)

            run_eval.append(eval_row)

            # Record parameters
            if config['train']['save_params']:
                param_dict = record_params(model, param_dict)      


        _, test_acc, test_auc, _, test_correct, _, _ = evaluate(device, model, test_pairs, test_labels, config)
        print("Test Accuracy:     %0.4f" % test_acc)
        test_accs.append(test_acc)
        test_aucs.append(test_auc)

        # Save the subject-FC ids so order can be replicated
        ids_dict[str(run)] = {
            'train': train_ids.tolist(),
            'valid': valid_ids.tolist(),
            'test': test_ids.tolist()
        }

        eval_df = pd.DataFrame(run_eval)
        eval_dfs.append(eval_df)

        # Generate evaluation and parameter plots
        plot_evaluations(eval_df, save_dir, run=run)

        param_start = time.time()
        if config['train']['save_params']:
            plot_params(param_dict, save_dir, run=run)
        print("Param plotting took %0.2f" % (time.time()-param_start))

        # Save distances
        train_dist_df = pd.DataFrame(train_distances)
        train_dist_df.insert(0, "Label", train_labels)
        train_dist_df.to_excel(save_dir+'/training_distances_.xlsx')
        valid_dist_df = pd.DataFrame(valid_distances)
        valid_dist_df.insert(0, "Label", valid_labels)
        valid_dist_df.to_excel(save_dir+'/validation_distances_.xlsx')

        # Save correct predictions as heatmap
        plot_classifications(train_pred, valid_pred, train_labels, valid_labels, test_correct, save_dir, run)

        # Save correct classification data
        train_class_dfs.append(pd.DataFrame(train_pred))
        valid_class_dfs.append(pd.DataFrame(valid_pred))
        test_class_dfs.append(pd.DataFrame(test_correct))

        # Plot confusion matrix data
        train_confusion = np.array(train_confusion)
        valid_confusion = np.array(valid_confusion)
        os.makedirs(save_dir+'/plots/confusion/'+str(run))
        plot_confusion(train_confusion, valid_confusion, save_dir, run)

        run_exec_time = time.time() - run_start
        print("This run took %.2f seconds" % run_exec_time)
        print("Expected to finish in %.2f seconds" % ((num_runs - run - 1)*run_exec_time))


    # Save the ids used in each run/fold
    with open('results/'+save_name+'/data_ids.json', 'w') as writer:
        json.dump(ids_dict, writer, indent=4)

    # Print and save the test accuracies
    test_acc_mean = np.mean(test_accs)
    test_acc_std = np.std(test_accs)

    cv_results_string = "Test Accuracy Mean: %3f    Test Accuracy Std: %3f" % (test_acc_mean,test_acc_std)
    print("\nCross Validation Results")
    print(cv_results_string)
    with open(save_dir+'/cross_validation_results.txt', 'w') as writer:
        for k, test_acc in enumerate(test_accs):
            writer.write("Fold %i Test Accuracy: %3f\n" % (k+1, test_acc))
        writer.write(cv_results_string)

    # save evaluation metrics
    with pd.ExcelWriter(save_dir+'/evaluation.xlsx') as writer:
        for k, df in enumerate(eval_dfs):
            df.to_excel(writer, sheet_name=str(k))

    with pd.ExcelWriter(save_dir+'/train_classifications.xlsx') as writer:
        for k, df in enumerate(train_class_dfs):
            df.to_excel(writer, sheet_name=str(k))

    with pd.ExcelWriter(save_dir+'/validation_classifications.xlsx') as writer:
        for k, df in enumerate(valid_class_dfs):
            df.to_excel(writer, sheet_name=str(k))

    with pd.ExcelWriter(save_dir+'/test_classifications.xlsx') as writer:
        for k, df in enumerate(test_class_dfs):
            df.to_excel(writer, sheet_name=str(k))

    print("Total Execution Time: %.2f" % (time.time() - start_time))