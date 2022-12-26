from ..egt_molgraph import EGT_MOL

class EGT_MOLCLINTOX(EGT_MOL):
    def __init__(self, **kwargs):
        super().__init__(output_dim=2, **kwargs)
