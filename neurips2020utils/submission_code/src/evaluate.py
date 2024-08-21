import argparse
import time
import os
import json
from datetime import datetime
import tempfile
import traceback

# import experiment_impact_tracker
# from experiment_impact_tracker import compute_tracker
# from experiment_impact_tracker.compute_tracker import ImpactTracker

from .submission_code import main as predictor
from .utils.metric_utils import compute_metric
from .utils.dataset import Nlc2CmdDS
from .utils.dataloaders import Nlc2CmdDL




# def compute_energyusage(annotation_filepath):
#
#     try:
#         tmplogdir = tempfile.mkdtemp()
#         print(f'Logging energy evaluation in directory: {tmplogdir}')
#
#         tracker = ImpactTracker(tmplogdir)
#         nlc2cmd_dl = get_dataloader(annotation_filepath)
#
#         tracker.launch_impact_monitor()
#         grnd_truth, _, _ = get_predictions(nlc2cmd_dl)
#         n = len(grnd_truth)
#
#         info = tracker.get_latest_info_and_check_for_errors()
#
#         tracker.p.terminate()
#         # experiment_impact_tracker.data_utils.log_final_info(tracker.logdir)
#
#         # stats = compute_tracker.read_latest_stats(tmplogdir)
#         # energy_watts = stats.get('rapl_estimated_attributable_power_draw', 0.0)
#         # energy_mwatts = (energy_watts * 1000.0) / n
#
#         # result = {
#         #     'status': 'success',
#         #     'energy_mwh': energy_mwatts
#         # }
#
#     except Exception as err:
#         result = {
#             'status': 'error',
#             'error_message': str(err),
#             'energy_mwh': 0.0
#         }
#
#         print(f'Exception occurred in energy consumption computation')
#         print(traceback.format_exc())
#
#     finally:
#         return result


if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.output_folderpath, exist_ok=True)

    if args.mode == 'eval':
        result = evaluate_model(args.annotation_filepath, args.params_filepath)
    # elif args.mode == 'energy':
    #     result = compute_energyusage(args.annotation_filepath)

    with open(os.path.join(args.output_folderpath, 'result.json'), 'w') as f:
        json.dump(result, f)
