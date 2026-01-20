import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copy import deepcopy
from collections import defaultdict
from augmentations.transforms_adacontrast import get_tta_transforms
import torch
import torch.nn as nn
import math

class SynTTA(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.episodic = cfg.MODEL.EPISODIC
        self.dataset_name = cfg.CORRUPTION.DATASET
        self.steps = cfg.OPTIM.STEPS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        self.mem = CSTU(capacity=cfg.SYNTTA.MEMORY_SIZE, num_class=num_classes, lambda_t=cfg.SYNTTA.LAMBDA_T, lambda_u=cfg.SYNTTA.LAMBDA_U)
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
            momentum_bn = NewBN(bn_layer, self.cfg.SYNTTA.GAMMA)
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

    @staticmethod
    def build_ema(model):
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model

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

class CSTU:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u

        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy

    def per_class_dist(self):
        per_class_occupied = [0] * self.num_class
        for cls, class_list in enumerate(self.data):
            per_class_occupied[cls] = len(class_list)

        return per_class_occupied

    def add_instance(self, instance):
        assert (len(instance) == 3)
        x, prediction, uncertainty = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls, score):
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes()
                return self.remove_from_classes(majority_classes, score)
        else:
            return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes: list[int], score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                uncertainty = item.uncertainty
                age = item.age
                score = self.heuristic_score(age=age, uncertainty=uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls

        if max_class is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            else:
                return False
        else:
            return True

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied == max_occupied:
                classes.append(i)

        return classes

    def heuristic_score(self, age, uncertainty):
        return self.lambda_t * 1 / (1 + math.exp(-age / self.capacity)) + self.lambda_u * uncertainty / math.log(self.num_class)

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return

    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)

        tmp_age = [x / self.capacity for x in tmp_age]

        return tmp_data, tmp_age

    
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



