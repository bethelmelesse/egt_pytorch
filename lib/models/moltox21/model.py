from ..egt_molgraph import EGT_MOL

class EGT_MOLTOX21(EGT_MOL):
    def __init__(self, **kwargs):
        super().__init__(output_dim=12, **kwargs)
