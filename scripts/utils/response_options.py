import argparse


def parse_bool(value):
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected one of: 0, 1, true, false, yes, no, on, off")


def add_save_model_input_arg(parser):
    parser.add_argument(
        "--save-model-input",
        "--save_model_input",
        type=parse_bool,
        default=False,
        help="Save prompt messages in response output. Default: 0.",
    )
