import torch
import torch.nn.functional as F

from lib.training.training import cached_property
from ..egt_mol_training import EGT_MOL_Training

from lib.models.moltox21 import EGT_MOLTOX21
from lib.data.moltox21 import MOLTOX21StructuralSVDGraphDataset

class MOLTOX21_Training(EGT_MOL_Training):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name    = 'moltox21',
            dataset_path    = 'cache_data/MOLTOX21',
            evaluation_type = 'prediction',
            predict_on      = ['test'],
            state_file      = None,
        )
        return config_dict
    
    def get_dataset_config(self):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, MOLTOX21StructuralSVDGraphDataset
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, EGT_MOLTOX21
    
    def calculate_bce_loss(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        targets_not_nan = torch.logical_not(torch.isnan(targets))
        targets_final = targets[targets_not_nan]
        outputs_final = outputs[targets_not_nan]
        return F.binary_cross_entropy_with_logits(outputs_final, targets_final)
    
    def calculate_loss(self, outputs, inputs):
        return self.calculate_bce_loss(outputs, inputs['target'])

    @cached_property
    def evaluator(self):
        from ogb.graphproppred import Evaluator
        evaluator = Evaluator(name = "ogbg-moltox21")
        return evaluator
    
    def prediction_step(self, batch):
        return dict(
            predictions = torch.sigmoid(self.model(batch)),
            targets     = batch['target'],
        )
        
    def evaluate_predictions(self, predictions):
        input_dict = {"y_true": predictions['targets'], 
                      "y_pred": predictions['predictions']}
        results = self.evaluator.eval(input_dict)
        
        xent = self.calculate_bce_loss(torch.from_numpy(predictions['predictions']),
                                       torch.from_numpy(predictions['targets'])).item()
        results['xent'] = xent
        
        for k, v in results.items():
            if hasattr(v, 'tolist'):
                results[k] = v.tolist()
        return results
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        print(f'Evaluating on {dataset_name}')
        results = self.evaluate_predictions(predictions)
        return results
        
SCHEME = MOLTOX21_Training
