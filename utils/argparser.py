import argparse

parser = argparse.ArgumentParser(prog='NL2BashCMD-NextGen')

parser.add_argument('-k', '--together_api_key', required=False)
parser.add_argument('-d', '--dataset_path', required=False)
parser.add_argument('-p', '--proxy_pia', required=False, default='')
parser.add_argument('-u', '--url', required=False, help='server url to which send requests')
parser.add_argument('-r', '--rate_limit', required=False, help='rate limit to send requests', default=1, type=float)
parser.add_argument('-m', '--model_path', required=False, help='model path or url to use')
parser.add_argument('-n', '--notes', help='additional notes', default='')
parser.add_argument('-o', '--output_path', required=False, help='output path to store the model\'s outputs')
