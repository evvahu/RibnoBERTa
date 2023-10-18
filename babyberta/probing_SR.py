import sys
import os
import torch
import logging
import toml
print(os.getcwd())
sys.path.append('/home/evhuber/data/RibnoBERTa')
from torch.utils.data import DataLoader
from SR_classification.main_classify import train_neural, evaluate
from SR_classification.data import DataProvider, create_label_encoder
from SR_classification.SR_utils import read_data_splits_ids
from SR_classification.neural_classifier import SimpleSRClassifier
# surprisal: true vs. false



def do_probing_SR(config_path, data_path, cl_model_path, lm_model_path, results_path, log_path, training=True):

    print('load config') 
    config = toml.load(config_path)
    print('config loaded')
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
                                                  logging.FileHandler(log_path)])
    logging.info(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    splits = read_data_splits_ids(data_path, label_column=config['label_column'])
    data_train = splits['train']
    data_valid = splits['valid']
    data_test = splits['test']

    all_labels = data_train[4] + data_valid[4] + data_test[4]
    label_nr = len(set(all_labels))
    label_encoder = create_label_encoder(all_labels)

    print('loading train data ...')
    train_data = DataProvider(data_train[3], data_train[1], data_train[2], data_train[4],label_encoder, lm_model_path, batch_size = config['batch_size'], max_len=config['max_len'], lower_case=config['lower_case'], max_pooling=config['max_pooling'])
    print('loading valid data ...')
    valid_data = DataProvider(data_valid[3], data_valid[1], data_valid[2], data_valid[4], label_encoder, lm_model_path, batch_size = config['batch_size'], max_len=config['max_len'], lower_case=config['lower_case'], max_pooling=config['max_pooling'])
    print('loading test data ...')
    test_data = DataProvider(data_test[3], data_test[1], data_test[2], data_test[4], label_encoder, lm_model_path, batch_size = config['batch_size'], max_len=config['max_len'], lower_case=config['lower_case'], max_pooling=config['max_pooling']) 
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=False, num_workers = 0)
    valid_loader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=False, num_workers = 0)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers = 0)
    if training:
        logging.info('starting to train classifier model')
        logging.info('Building model') 
        model =  SimpleSRClassifier(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'], label_nr=label_nr, dropout_rate= config['dropout_rate'], device=device) #self, input_dim, hidden_dim, label_nr, dropout_rate, device='cpu'):
        logging.info('Start training...')
        if not config['evaluate_at_epoch_end']:
            train_neural(model, device, train_loader, valid_loader,config, cl_model_path, label_encoder)
        else:
            train_neural(model, device, train_loader, valid_loader,config, cl_model_path, label_encoder, result_path=results_path, data_test=data_test, test_loader = test_loader)

    logging.info('Start evaluating...')
    evaluate(config, cl_model_path, device, test_loader, data_test[0], label_encoder, results_path)
    

        


