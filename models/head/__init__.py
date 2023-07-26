from .pa_head import PA_Head
# from .psenet_head import PSENet_Head
from .builder import build_head
# for PAN++
from .CoText_det_head import CoText_DetHead
from .pan_pp_rec_head import PAN_PP_RecHead
from .cotext_rec_head_ctc import CoText_RecHead_CTC
from .cotext_track_head import CoText_TrackHead

__all__ = ['PA_Head','CoText_DetHead', 'PAN_PP_RecHead', 'CoText_RecHead_CTC', 'CoText_TrackHead']
