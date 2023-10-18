import torch


def prepare_device():
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device
