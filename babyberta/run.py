
from params import Params
from job import main
import argparse
import os
# list of corpora (ordered)
# nr of epochs per corpora

currentparams = {
    # data
    'sample_with_replacement': False,  # this must be False if corpus order is to be preserved during training
    'training_order': 'original',  # original or shuffled, use this alongside consecutive_masking=True
    'consecutive_masking': False,  # better dev pp and grammatical accuracy when false
    'num_sentences_per_input': 1,  # if too large -> may exceed CUDA memory, 1 is best for good number-agreement
    'include_punctuation': True,
    'allow_truncated_sentences': False,
    'num_mask_patterns': 10,  # 10 is better than fewer
    'mask_pattern_size': 2,  # used only if probabilistic_masking = False
    'probabilistic_masking': True,
    'mask_probability': 0.15,  # used only if probabilistic_masking = true
    'leave_unmasked_prob_start': 0.0,  # better performance if no unmasking
    'leave_unmasked_prob': 0.0,  # better performance if no unmasking
    'random_token_prob': 0.1,
    'corpora': (),
    'tokenizer': 'test_tokenizer',  # larger than 8k slightly reduces performance
    'add_prefix_space': True,  # better if True, whether to treat first token like any other token (False in GPT-2)
    'max_input_length': 128,  # unacceptable performance if lower than ~32

    # training
    'batch_size': 16,
    'lr': 1e-4,  # 1e-4 is used in fairseq (and performs better here), and 1e-3 is default in huggingface
    'num_epochs': 1,  # use 1 epoch to use dynamic masking
    'num_warmup_steps': 24_000,  # 24K used in Roberta-base
    'weight_decay': 0.0,

    # model
    'load_from_checkpoint': 'none',
    'hidden_size': 256,
    'num_layers': 8,
    'num_attention_heads': 8,
    'intermediate_size': 1024,
    'initializer_range': 0.02,  # stdev of trunc normal for initializing all weights
    'layer_norm_eps': 1e-5,  # 1e-5 default in fairseq (and slightly better performance), 1e-12 default in hgugingface,
    
    'project_path': 'test_project',
    'save_path': 'test_save',

    #probing SR
    'SR_config_path': 'config_SR.toml',
    'SR_data_path': 'set1',
    'SR_cl_model_path': 'results/test1',
    #'SR_lm_model_path':'results/test1/',
    'SR_log_path': 'results/test1',
    'SR_res_path': 'results/test1'


}
#config_path, data_path, cl_model_path, lm_model_path, results_path, log_path)

corpora = ['age_1_6', 'age_2_0', 'age_2_6', 'age_3_0', 'age_3_6', 'age_4_0', 'age_4_6', 'age_5_0'] #age_1_6.txt	age_2_0.txt	age_2_6.txt	age_3_0.txt	age_3_6.txt	age_4_0.txt	age_4_6.txt	age_5_0.txt
check_point = '' 
for i, corp in enumerate(corpora):
    print('RUN: {}'.format(str(i)))
    if i == 0:
        #load from checkpoint is none
        currentparams['corpora'] = (corp,)
        currentparams['save_path'] = os.path.join(currentparams['save_path'],'model_{}'.format(str(i)))
        main(currentparams)
       
        currentparams['load_from_checkpoint'] = currentparams['save_path']
        

    else:
        currentparams['save_path'] = currentparams['save_path'].replace('model_{}'.format(str(i-1)), 'model_{}'.format(str(i))) 
        print('SECOND PATH', currentparams['save_path'])
        currentparams['corpora'] = currentparams['corpora'] + (corp,)
        main(currentparams)
        currentparams['load_from_checkpoint'] = currentparams['save_path']
    
    

