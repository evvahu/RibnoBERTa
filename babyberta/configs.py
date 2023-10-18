from pathlib import Path
import os

class Dirs:
    root = Path(__name__).parent.parent.absolute() / 'test_project'
    #root = root / Path('test_project')
    data = root / 'data'
    corpora = data / 'corpora'
    tokenizers = data / 'tokenizers'
    saved_models = root / 'saved_models'

    # probing data can be found at https://github.com/phueb/Zorro/tree/master/sentences
    #probing_sentences = Path('/') / 'media' / 'ludwig_data' / 'Zorro' / 'sentences' / 'babyberta'
    probing = root / 'probing'
    SR_data = probing / 'SR'
    GR_data = probing / 'GR'
    probing_results = root / 'probing_results'
    probing_results_SR = probing_results / 'SR_res'
    probing_results_GR = probing_results / 'GR_res'

    # wikipedia sentences file was created using https://github.com/akb89/witokit
    #wikipedia_sentences = Path.home() / 'witokit_download_1' / 'processed.txt'


class Data:
    min_sentence_length = 3
    train_prob = 0.8  # probability that sentence is assigned to train split
    mask_symbol = '<mask>'
    pad_symbol = '<pad>'
    unk_symbol = '<unk>'
    bos_symbol = '<s>'
    eos_symbol = '</s>'
    roberta_symbols = [mask_symbol, pad_symbol, unk_symbol, bos_symbol, eos_symbol]


class Training:
    feedback_interval = 1000

    # for the published paper, we trained as many steps as needed to complete all epochs.
    # however, we only reported results of evaluations at a fixed checkpoint (e.g. 260k steps)
    max_step = None


class Eval:
    interval = 20_000
