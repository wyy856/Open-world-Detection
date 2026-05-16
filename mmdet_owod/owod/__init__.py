import importlib.util

from .wood_loss import WOODProposalLoss

if importlib.util.find_spec("mmdet") is not None and importlib.util.find_spec("mmengine") is not None:
    from .owod_bbox_head import OWODShared2FCBBoxHead
    from .owod_metric import OWODMetric
else:
    OWODShared2FCBBoxHead = None
    OWODMetric = None

__all__ = ["WOODProposalLoss", "OWODShared2FCBBoxHead", "OWODMetric"]
