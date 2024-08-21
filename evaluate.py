# evaluate the model's outputs based on submission code of neurips2020 competition
import argparse
import json
import time
from datetime import datetime

from neurips2020utils.compute_final_score import get_dataloader
from neurips2020utils.submission_code.src.utils.metric_utils import compute_metric
from neurips2020utils.submission_code.src.utils.dataloaders import Nlc2CmdDL
from neurips2020utils.submission_code.src.utils.dataset import Nlc2CmdDS


def predict(invocations, result_cnt=5):
    """
    Function called by the evaluation script to interface the participants model
    `predict` function accepts the natural language invocations as input, and returns
    the predicted commands along with confidences as output. For each invocation,
    `result_cnt` number of predicted commands are expected to be returned.

    Args:
        1. invocations : `list (str)` : list of `n_batch` (default 16) natural language invocations
        2. result_cnt : `int` : number of predicted commands to return for each invocation

    Returns:
        1. commands : `list [ list (str) ]` : a list of list of strings of shape (n_batch, result_cnt)
        2. confidences: `list[ list (float) ]` : confidences corresponding to the predicted commands
                                                 confidence values should be between 0.0 and 1.0.
                                                 Shape: (n_batch, result_cnt)
    """

    n_batch = len(invocations)

    # `commands` and `confidences` have shape (n_batch, result_cnt)
    commands = [
        [''] * result_cnt
        for _ in range(n_batch)
    ]
    confidences = [
        [1.0] * result_cnt
        for _ in range(n_batch)
    ]

    ################################################################################################
    #     Participants should add their codes to fill predict `commands` and `confidences` here    #
    ################################################################################################

    ################################################################################################
    #                               Participant code block ends                                    #
    ################################################################################################

    return commands, confidences


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--annotation_filepath', type=str, required=True)
    parser.add_argument('--params_filepath', type=str, required=True)
    parser.add_argument('--output_folderpath', type=str, required=True)
    parser.add_argument('--mode', type=str, required=False, default='eval')

    return parser


def get_dataloader(annotation_filepath):
    nlc2cmd_ds = Nlc2CmdDS(annotation_filepath)
    nlc2cmd_dl = Nlc2CmdDL(nlc2cmd_ds, batchsize=8, shuffle=True)
    return iter(nlc2cmd_dl)


def get_params(params_filepath):
    # with open(params_filepath, 'r') as f:
    #     params = json.load(f)
    return {'u1':1.0,'u2':1.0} # official params, see ./neurips2020utils/submission_code/configs/core/evaluation_params.json


def validate_predictions(predicted_cmds, predicted_confds, n_batch, result_cnt):
    assert len(predicted_cmds) == n_batch, \
        f'{len(predicted_cmds)} commands predicted for {n_batch} invocations'

    assert len(predicted_confds) == n_batch, \
        f'{len(predicted_confds)} confidences predicted for {n_batch} invocations'

    for i in range(n_batch):
        assert 1 <= len(predicted_cmds[i]) <= result_cnt, \
            f'{len(predicted_cmds[i])} commands predicted for an invocations. Expected between 1 and {result_cnt}'

        assert 1 <= len(predicted_confds[i]) <= result_cnt, \
            f'{len(predicted_confds[i])} confidences predicted for an invocations. Expected between 1 and {result_cnt}'

        assert not (False in [0.0 <= x <= 1.0 for x in predicted_confds[i]]), \
            f'Confidence value beyond the allowed range of [0.0, 1.0] found in predictions'


def get_predictions(nlc2cmd_dl):
    result_cnt = 5
    i = 0
    ground_truths = []
    predicted_cmds, predicted_confds = [], []

    for invocations, cmds in nlc2cmd_dl:
        batch_predicted_cmds, batch_predicted_confd = predict(invocations, result_cnt=result_cnt)
        validate_predictions(batch_predicted_cmds, batch_predicted_confd, len(invocations), result_cnt)

        ground_truths.extend(cmds)
        predicted_cmds.extend(batch_predicted_cmds)
        predicted_confds.extend(batch_predicted_confd)

        if i % 15 == 0:
            now = datetime.now().strftime('%d/%m %H:%M:%S')
            print(f'\t{now} :: {i} batches predicted')
        i += 1

    return ground_truths, predicted_cmds, predicted_confds


def get_score(prediction_scores):
    score = -1.0
    if len(prediction_scores) == 0:
        return score

    has_positive_score = True in [x > 0 for x in prediction_scores]

    if has_positive_score:
        score = max(prediction_scores)
    else:
        score = sum(prediction_scores) / float(len(prediction_scores))

    return score


def compute_score(ground_truths, predicted_cmds, predicted_confds, metric_params):
    prediction_scores = []

    for grnd_truth_cmd in ground_truths:
        for i, predicted_cmd in enumerate(predicted_cmds):

            if predicted_cmd is None or len(predicted_cmd) == 0:
                continue

            predicted_confidence = predicted_confds[i]
            pair_score = compute_metric(predicted_cmd, predicted_confidence, grnd_truth_cmd, metric_params)
            prediction_scores.append(pair_score)

    score = get_score(prediction_scores)

    # print('-' * 50)
    # print(f'Ground truth: {ground_truths}')
    # print(f'Predictions: {predicted_cmds}')
    # print(f'Score: {score}')

    return score


def evaluate_model(annotation_filepath, params_filepath):
    try:
        params = get_params(params_filepath)

        nlc2cmd_dl = get_dataloader('./neurips2020utils/submission_code/configs/annotations/local_eval_annotations.json') # TODO
        # # # TODO scrivere il nostro dataloader custom
        # #
        # stime = time.time()
        fn_return = get_predictions(nlc2cmd_dl) # TODO
        # # # TODO scrivere il nostro get_predictions o rendere il json di output compatibile
        # #
        # total_time_taken = time.time() - stime
        # #
        ground_truths, predicted_cmds, predicted_confds = fn_return # TODO vedi sopra
        # n = len(ground_truths)
        n = 1

        print('----------------------- Predictions -----------------------')
        # TODO SAMPLE EXAMPLES
        '''		
        find Path -name Regex | xargs -I {} rm {}
		find Path -type f -exec rm {} \;
		rm -r -f File
		find Path -type f -print0 | xargs -0 -I {} rm {}
		find Path -exec rm {} \;
		'''
        ground_truths, predicted_cmds, predicted_confds = [['rm -r -f File']], [['ls']], [['0.237613']]

        scores = [
            compute_score(ground_truths[i], predicted_cmds[i], predicted_confds[i], params)
            for i in range(n)
        ]

        # print(f'sum: {sum(scores)}, n: {n}')
        # print('----------------------- Predictions -----------------------')

        mean_score = sum(scores) / float(n)
        time_taken = 1.0 # total_time_taken / float(n)

        result = {
            'status': 'success',
            'time_taken': time_taken,
            'score': mean_score
        }

    except Exception as err:
        result = {
            'status': 'error',
            'error_message': str(err)
        }

    return result


r = evaluate_model('annotation_file_path', './neurips2020utils/metric_params.json')

print(r)
