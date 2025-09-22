from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn


class SynTTA(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.episodic = cfg.MODEL.EPISODIC
        self.dataset_name = cfg.CORRUPTION.DATASET
        self.steps = cfg.OPTIM.STEPS
        assert self.steps > 0, "requires >= 1 step(s) to forward and update"

        # configure model and optimizer
        self.configure_model()
        self.params, param_names = self.collect_params()
        self.optimizer = self.setup_optimizer() if len(self.params) > 0 else None
        self.print_amount_trainable_params()


        self.input_buffer = None
        self.window_length = cfg.TEST.WINDOW_LENGTH
        self.pointer = torch.tensor([0], dtype=torch.long).cuda()

        self.has_bn = any([isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules()])

        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        # follow the Self-Training setting in RoTTA https://arxiv.org/abs/2303.13899
        self.mem = ClassBalanceBank(capacity=cfg.SYNTTA.MEMORY_SIZE, num_class=num_classes, lambda_t=cfg.SYNTTA.LAMBDA_T, lambda_u=cfg.SYNTTA.LAMBDA_U)
        self.model_ema = self.build_ema(self.model)
        self.transform = get_tta_transforms(cfg)
        self.nu = cfg.SYNTTA.NU
        self.update_frequency = cfg.SYNTTA.UPDATE_FREQUENCY
        self.current_instance = 0

        self.eta = cfg.SYNTTA.ETA # OBPC forgetting factor \eta
        self.alpha = torch.ones(self.num_classes, device=self.device)  # OBPC initial Prior Counts

        self.src_model = deepcopy(self.model).cpu()
        for param in self.src_model.parameters():
            param.detach_()

    def forward(self, x):
        if self.episodic:
            self.reset()

        x = x if isinstance(x, list) else [x]

        if x[0].shape[0] == 1:  # single sample test-time adaptation
            # create the sliding window input
            if self.input_buffer is None:
                self.input_buffer = [x_item for x_item in x]
                # set bn1d layers into eval mode, since no statistics can be extracted from 1 sample
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=False)
            elif self.input_buffer[0].shape[0] < self.window_length:
                self.input_buffer = [torch.cat([self.input_buffer[i], x_item], dim=0) for i, x_item in enumerate(x)]
                # set bn1d layers into train mode
                self.change_mode_of_batchnorm1d(self.models, to_train_mode=True)
            else:
                for i, x_item in enumerate(x):
                    self.input_buffer[i][self.pointer] = x_item

            if self.pointer == (self.window_length - 1):
                # update the model, since the complete buffer has changed
                for _ in range(self.steps):
                    outputs = self.forward_and_adapt(self.input_buffer)
                outputs = outputs[self.pointer.long()]
            else:
                # create the prediction without updating the model
                if self.has_bn:
                    # forward the whole buffer to get good batchnorm statistics
                    outputs = self.forward_sliding_window(self.input_buffer)
                    outputs = outputs[self.pointer.long()]
                else:
                    # only forward the current test sample, since there are no batchnorm layers
                    outputs = self.forward_sliding_window(x)

            # increase the pointer
            self.pointer += 1
            self.pointer %= self.window_length

        else:   # common batch adaptation setting
            for _ in range(self.steps):
                outputs = self.forward_and_adapt(x)

        return outputs

    """
    Online Bayesian Prior Correction (OBPC) Module
    """
    @torch.enable_grad()
    def apply_online_bayesian_prior_correction(self, ema_out):
        batch_size = ema_out.shape[0]
        batch_prior = ema_out.softmax(1).mean(0)
        counts = batch_size * batch_prior
        self.alpha = (1 - self.eta) * self.alpha + counts
        robust_prior = self.alpha / self.alpha.sum()
        smooth = max(1 / ema_out.shape[0], 1 / ema_out.shape[1]) / torch.max(robust_prior)
        robust_prior = (robust_prior + smooth) / (1 + smooth * ema_out.shape[1])
        return ema_out * robust_prior

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        x = x[0]
        # batch data
        with torch.no_grad():
            self.model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(x)
            ema_out = self.apply_online_bayesian_prior_correction(ema_out)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # add into memory
        for i, data in enumerate(x):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model()

        self.model = ema_update_model(
            model_to_update=self.model,
            model_to_merge=self.src_model,
            momentum=0.99,
            device=self.device
        )
        return ema_out

    def update_model(self,):
        self.model.train()
        self.model_ema.train()

        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = self.model(strong_sup_aug)
            instance_weight = timeliness_reweighting(ages)
            l_sup = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        if l is not None:
            l.backward()
            self.optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.nu)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def configure_model(self):

        self.model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in self.model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(self.model, name)
            if isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = GMBN2d
            else:
                raise RuntimeError()
            momentum_bn = NewBN(bn_layer, self.cfg.SYNTTA.ALPHA, self.cfg.SYNTTA.GAMMA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(self.model, name, momentum_bn)
        return self.model

    @torch.no_grad()
    def forward_sliding_window(self, x):
        imgs_test = x[0]
        return self.model(imgs_test)

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def setup_optimizer(self):
        return torch.optim.Adam(self.params,
                                lr=self.cfg.OPTIM.LR,
                                betas=(self.cfg.OPTIM.BETA, 0.999),
                                weight_decay=self.cfg.OPTIM.WD)


    def print_amount_trainable_params(self):
        trainable = sum(p.numel() for p in self.params) if len(self.params) > 0 else 0
        total = sum(p.numel() for p in self.model.parameters())
        print(f"trainable/total parameters: {trainable}/{total} ({100 * trainable / total:.2f}%)")  

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
            
        self.optimizer = self.setup_optimizer()
        self.optimizer.load_state_dict(self.optimizer_state)

    @staticmethod
    def copy_model(model):
        coppied_model = deepcopy(model)
        return coppied_model

class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum: float):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum

        # Deepcopy source statistics and affine parameters from the provided layer
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))
        
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        # Initialize target statistics with the source statistics
        self.register_buffer("target_mean", deepcopy(bn_layer.running_mean))
        self.register_buffer("target_var", deepcopy(bn_layer.running_var))
        self.eps = bn_layer.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class GMBN2d(MomentumBN):
    """
    Gradient-Modulated Batch Normalization (2D).

    Inherits initialization logic from MomentumBN and implements the adaptive
    forward pass as described in the SynTTA paper.
    """
    def __init__(self, bn_layer: nn.BatchNorm2d, gamma: float = 1.0):
        """
        Args:
            bn_layer (nn.BatchNorm2d): The pre-trained source BatchNorm2d layer.
            gamma (float): The adaptive scaling factor for the suppression signal (\gamma).
        """
        super().__init__(bn_layer, momentum=bn_layer.momentum)
        self.gamma = gamma

        # Buffers for online min-max tracking of the gradient norm
        self.register_buffer('min_grad_norm', torch.full((self.num_features,), float('inf')))
        self.register_buffer('max_grad_norm', torch.full((self.num_features,), float('-inf')))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:

            b_mean = x.mean([0, 2, 3])
            b_var = x.var([0, 2, 3], unbiased=False)
   
            ema_mean = (1 - self.momentum) * self.target_mean + self.momentum * b_mean
            ema_var = (1 - self.momentum) * self.target_var + self.momentum * b_var

            with torch.no_grad():

                g_c = torch.abs(ema_mean - self.source_mean)
                
                self.min_grad_norm.copy_(torch.min(self.min_grad_norm, g_c))
                self.max_grad_norm.copy_(torch.max(self.max_grad_norm, g_c))

                norm_range = self.max_grad_norm - self.min_grad_norm
                g_norm_c = (g_c - self.min_grad_norm) / (norm_range + self.eps)
                g_norm_c = torch.clamp(g_norm_c, 0.0, 1.0)
                
                s_c = self.momentum * (1 + self.gamma * g_norm_c)
                s_c = torch.clamp(s_c, 0.0, 1.0) 

            s_c = s_c.view(-1)
            final_mean = s_c * self.source_mean + (1 - s_c) * ema_mean
            final_var = s_c * self.source_var + (1 - s_c) * ema_var
            
            self.target_mean.copy_(final_mean.detach())
            self.target_var.copy_(final_var.detach())
            
            mean_to_use, var_to_use = final_mean, final_var
        else:
            mean_to_use, var_to_use = self.target_mean, self.target_var

        mean = mean_to_use.view(1, -1, 1, 1)
        var = var_to_use.view(1, -1, 1, 1)
        
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        
        return x_normalized * weight + bias

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def is_empty(self):
        return self.cnt == 0
    def reset(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0
    def update(self, val, n=1):
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum / self.cnt

@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, device, update_all=False):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update

class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):

        if not self.is_empty():
            self.age += 1

    def get_details(self):

        return self.data, self.uncertainty, self.age

    def is_empty(self):

        return self.data is None

class ClassBalanceBank:

    def __init__(self, capacity: int, num_class: int, alpha: float = 1.0, beta: float = 1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.alpha = alpha  # Weight for the uncertainty component
        self.beta = beta    # Weight for the class balance component
        self.per_class_limit = max(1, capacity // num_class)
        self.bank: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]

    def get_occupancy(self) -> int:
        return sum(len(class_list) for class_list in self.bank)

    def get_class_distribution(self) -> list[int]:
        return [len(class_list) for class_list in self.bank]

    def push(self, sample: tuple):
        assert len(sample) == 3
        data, prediction, uncertainty = sample
        
        new_item = MemoryItem(data=data, uncertainty=uncertainty, age=0)
        # Calculate the eviction score for the new sample
        new_score = self.calculate_eviction_score(uncertainty, prediction)

        if self._find_and_prepare_slot(prediction, new_score):
            self.bank[prediction].append(new_item)
        
        # Increment the age of all existing items in the bank
        self._increment_ages()

    def _find_and_prepare_slot(self, class_idx: int, new_score: float) -> bool:
        class_list = self.bank[class_idx]
        class_occupancy = len(class_list)
        total_occupancy = self.get_occupancy()

        # Case 1: The target class is not yet full to its limit.
        if class_occupancy < self.per_class_limit:
            # Subcase 1a: The entire bank is not full, so no eviction is needed.
            if total_occupancy < self.capacity:
                return True
            # Subcase 1b: The bank is full, so a sample must be evicted from the most populated class.
            else:
                majority_class_indices = self._get_majority_class_indices()
                return self._evict_worst_sample(majority_class_indices, new_score)
        # Case 2: The target class is full, so a sample must be evicted from this class.
        else:
            return self._evict_worst_sample([class_idx], new_score)

    def _evict_worst_sample(self, candidate_classes: list[int], new_score: float) -> bool:
        eviction_candidate = {'class_idx': None, 'sample_idx': None, 'score': -1}

        for class_idx in candidate_classes:
            for sample_idx, item in enumerate(self.bank[class_idx]):
                # Calculate the eviction score for an existing sample
                score = self.calculate_eviction_score(item.uncertainty, class_idx)
                if score >= eviction_candidate['score']:
                    eviction_candidate.update({
                        'class_idx': class_idx, 
                        'sample_idx': sample_idx, 
                        'score': score
                    })
        
        # If no candidate was found (e.g., candidate_classes was empty), allow adding.
        if eviction_candidate['class_idx'] is None:
            return True

        # If the worst sample in the bank has a higher score than the new sample, evict it.
        if eviction_candidate['score'] > new_score:
            self.bank[eviction_candidate['class_idx']].pop(eviction_candidate['sample_idx'])
            return True
        # Otherwise, the new sample is not 'good' enough to be added.
        else:
            return False

    def _get_majority_class_indices(self) -> list[int]:
        class_dist = self.get_class_distribution()
        if not any(class_dist):
            return []
        max_occupancy = max(class_dist)
        return [i for i, occupancy in enumerate(class_dist) if occupancy == max_occupancy]

    def calculate_eviction_score(self, uncertainty: float, class_idx: int) -> float:
        total_occupancy = self.get_occupancy()
        if total_occupancy == 0:
            return self.alpha * uncertainty

        class_occupancy = len(self.bank[class_idx])
        
        balance_factor = class_occupancy / total_occupancy
        
        score = (self.alpha * uncertainty) + (self.beta * balance_factor)
        return score

    def _increment_ages(self):
        for class_list in self.bank:
            for item in class_list:
                item.increase_age()

    def get_all_samples(self) -> tuple[list, list]:
        all_data = []
        all_ages = []

        for class_list in self.bank:
            for item in class_list:
                all_data.append(item.data)
                all_ages.append(item.age)
        
        if self.capacity > 0:
            normalized_ages = [age / self.capacity for age in all_ages]
        else:
            normalized_ages = [0] * len(all_ages)

        return all_data, normalized_ages
    
def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))

@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module

def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)



