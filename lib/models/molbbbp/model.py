from ..egt_molgraph import EGT_MOL

class EGT_MOLBBBP(EGT_MOL):
    def __init__(self, **kwargs):
        super().__init__(output_dim=1, **kwargs)
