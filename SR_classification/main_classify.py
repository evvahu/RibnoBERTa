import argparse
import toml
import torch
from SR_classification.evaluate import get_accuracy
from SR_classification.data import DataProvider, create_label_encoder
from SR_classification.neural_classifier import SimpleSRClassifier
from torch import optim
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import random


def train_neural(model, device, train_loader, valid_loader, config, model_path, label_encoder, test_loader = False, data_test = False,result_path = False):

    if config['evaluate_at_epoch_end']:
        assert test_loader and result_path
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    current_patience = 0
    tolerance = 1e-5
    lowest_loss = float("inf")
    epoch = 1
    train_loss = 0.0
    best_f1 = 0.0
    total_train_losses = []
    total_val_losses = []
    entr_loss = nn.CrossEntropyLoss()
    for epoch in range(1, config['num_epochs']+1):
        train_losses = []
        valid_losses = []
        valid_accuracies = []
        valid_f1_scores = []

        model.train()
        for batch in train_loader:
            out = model(batch).squeeze().to("cpu")
            #loss = binary_class_cross_entropy(out, batch["l"])
            loss = entr_loss(out, torch.from_numpy(np.array(batch["l"])))
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        for batch in valid_loader:
            batch["device"] = device
            out = model(batch).squeeze().to("cpu")
            #predictions = convert_logits_to_binary_predictions(out)
            predictions = torch.argmax(out, dim=1)
            #loss = binary_class_cross_entropy(out, batch["l"].float())
            loss = entr_loss(out, torch.from_numpy(np.array(batch["l"])))
            valid_losses.append(loss.item())
            
            _, accur = get_accuracy(predictions, batch["l"])
            if config['label_nr'] > 2:
                f1 = f1_score(y_true=batch["l"], y_pred=predictions, average='macro')
            else:
                f1 = f1_score(y_true=batch["l"], y_pred=predictions, average='binary')
            valid_accuracies.append(accur)
            valid_f1_scores.append(f1)
        

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        valid_accuracy = np.average(valid_accuracies)
        valid_f1 = np.average(valid_f1_scores)
        total_val_losses.append(valid_loss)
        total_train_losses.append(train_loss)

        if config['early_stopping'] == "f1":
            if valid_f1 > best_f1 - tolerance:
                lowest_loss = valid_loss
                best_f1 = valid_f1
                current_patience = 0
                #torch.save(model.state_dict(), config['model_path'])
                torch.save(model, model_path)
                logging.info('at epoch {0:.3g}: valid_f1:{1:.3g}, best_f1:{2:.3g}, train loss: {3:.3g}, valid loss: {4:.3g}, valid accuracy: {5:.3g}'.format(epoch, valid_f1, best_f1, train_loss, valid_loss, valid_accuracy))
                if config['evaluate_at_epoch_end']:
                    out_path = result_path.replace('.txt', '_epoch{}.txt'.format(epoch))
                    evaluate(config, model_path, device, test_loader, data_test[0], label_encoder, out_path)
            else:
                if config['evaluate_at_epoch_end']:
                    out_path = result_path.replace('.txt', '_epoch{}.txt'.format(epoch))
                    evaluate(config, model_path, device, test_loader, data_test[0], label_encoder, out_path)
                current_patience += 1
                logging.info('at epoch {0:.3g}: valid_f1:{1:.3g}, best_f1:{2:.3g}, train loss: {3:.3g}, valid loss: {4:.3g}, valid accuracy: {5:.3g}'.format(epoch, valid_f1, best_f1, train_loss, valid_loss, valid_accuracy))
        # stop when loss is the lowest
        else:
            if lowest_loss - valid_loss > tolerance:
                lowest_loss = valid_loss
                best_epoch = epoch
                best_accuracy = valid_accuracy
                best_f1 = valid_f1
                current_patience = 0
                #torch.save(model.state_dict(), config['model_path'])
                torch.save(model, model_path)
                logging.info('at epoch {0:.3g}: valid_accuracy:{1:.3g}, best_accuracy:{2:.3g}, train loss: {3:.3g}, valid loss: {4:.3g}, lowest loss: {5:.3g}'.format(epoch, valid_f1, best_f1, train_loss, valid_loss, lowest_loss))
                if config['evaluate_at_epoch_end']: 
                    out_path = result_path.replace('.txt', '_epoch{}.txt'.format(epoch))
                    evaluate(config, model_path, device, test_loader, data_test[0], label_encoder, out_path)
            else:
                if config['evaluate_at_epoch_end']:
                    out_path = result_path.replace('.txt', '_epoch{}.txt'.format(epoch))
                    evaluate(config, model_path, device, test_loader, data_test[0], label_encoder, out_path)
                current_patience += 1
                logging.info('at epoch {0:.3g}: valid_accuracy:{1:.3g}, best_accuracy:{2:.3g}, train loss: {3:.3g}, valid loss: {4:.3g}, lowest loss: {5:.3g}'.format(epoch, valid_f1, best_f1, train_loss, valid_loss, lowest_loss))

        if current_patience > config['patience']:
            break

        #print("train loss: {}".format(train_loss))
        #print("best accuracy: {}, best f1: {}, best epoch: {}".format(best_accuracy, best_f1, best_epoch))

def evaluate(config,model_path, device, test_loader, info, label_encoder, result_path):
    model = torch.load(model_path)
    test_accuracies = []
    test_f1_scores = []
    all_predictions = []
    all_probabilities = []
    all_labels = []
    model.eval()
    for batch in test_loader:
        batch["device"] = device
        out = model(batch).squeeze().to("cpu")
        predictions = torch.argmax(out, dim=1)
        ids = predictions.long().view(-1,1)
        probabilities =  out.gather(1, ids)
        for i,l in enumerate(predictions.tolist()):
            all_predictions.append(l)
            all_probabilities.append(probabilities[i])
        for l in batch["l"].tolist():
            all_labels.append(l) 
        test_accuracies.append(get_accuracy(predictions, batch["l"])[1])
        if config['label_nr'] > 2:
            test_f1_scores.append(f1_score(batch["l"], predictions, average='macro'))
        else:
            f1 = f1_score(y_true=all_labels, y_pred=all_predictions, average='binary')
    _, accur = get_accuracy(all_predictions, all_labels)
    if config['label_nr'] > 2:
        f1 = f1_score(y_true=all_labels, y_pred=all_predictions, average='macro')
    else:
        f1 = f1_score(y_true=all_labels, y_pred=all_predictions, average='macro')
    #logging.info('evaluation on test data: accuracy: {0:.3g}, f1 score:{1:0.3g}'.format(np.round(np.mean(test_accuracies),2), np.round(np.mean(test_f1_scores),2)))
    logging.info('evaluation on test data: accuracy: {0:.3g}, f1 score:{1:0.3g}'.format(accur, f1)) 
    #print('accuracy: {0:.3g}, f1 score:{1:3g}, precision: {2:.3g}, recall: {3:.3g}'.format(np.mean(test_accuracies), np.mean(test_f1_scores), precision_score(all_labels, all_predictions), recall_score(all_labels, all_predictions)))    
    i = 0
    if len(result_path) > 0:
        with open(result_path, 'w') as wf:
            wf.write('{}\t{}\t{}\t{}\t{}\n'.format('sent_id', 'predicted', 'actual_role', 'same', 'probability'))
            for pred, labs, prob in zip(all_predictions, all_labels, all_probabilities):
                inf = info[i]
                i += 1
                wf.write('{}\t{}\t{}\t{}\t{}\n'.format(inf, label_encoder.inverse_transform([pred])[0], label_encoder.inverse_transform([labs])[0], pred==labs, prob)) 


