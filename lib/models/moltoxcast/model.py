from ..egt_molgraph import EGT_MOL

class EGT_MOLTOXCAST(EGT_MOL):
    def __init__(self, **kwargs):
        super().__init__(output_dim=617, **kwargs)
