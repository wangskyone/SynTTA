# coding=utf-8
import os
import sys 
sys.path.append(f"{os.getcwd()}/datasets")
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
from augmentations.transforms_adacontrast import rgb_loader, l_loader, image_train, image_test
from conf import complete_data_dir_path
from classification.utils.utils import get_source_target_names

class ImageDataset(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name,
                 transform=None, target_transform=None, indices=None, mode='Default', samples=None):
        """
        Initialize an image dataset.

        This class can be initialized in two ways:
        1. Default: By scanning a `root_dir` for images using domain names.
        2. From a list: By passing a pre-filtered list of `samples`.

        Args:
            dataset (str): Dataset name.
            task (str): Task type ('DG', 'DA', etc.).
            root_dir (str): Root directory containing the data.
            domain_name (str or list): Domain name(s).
            transform (callable, optional): Image transformations.
            target_transform (callable, optional): Target transformations.
            indices (list, optional): Specific indices to use. If None, use all.
            mode (str): Image loading mode ('Default', 'RGB', 'L').
            samples (list, optional): A list of samples, where each sample is a tuple
                                      of (image_path, class_label, domain_label).
                                      If provided, the dataset is initialized from this
                                      list, and `root_dir`/`domain_name` scanning is skipped.
        """
        self.task = task
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

        if samples is not None:
            # --- Initialization from a pre-loaded list of samples ---
            self.imgs = [(path, cls_label) for path, cls_label, _ in samples]
            all_labels = [cls_label for _, cls_label, _ in samples]
            all_domain_labels = [domain_label for _, _, domain_label in samples]

            self.labels = np.array(all_labels)
            self.domain_labels = np.array(all_domain_labels)
            self.x = [path for path, _, _ in samples]
            
            unique_domains = sorted(list(set(self.domain_labels)))
            self.domain_num = len(unique_domains)
            self.domain_indices = {domain_id: [] for domain_id in unique_domains}
            for i, domain_id in enumerate(self.domain_labels):
                self.domain_indices[domain_id].append(i)

        else:
            # --- Original logic: Initialization by scanning directories ---
            if isinstance(domain_name, list):
                self.imgs = []
                domain_labels_list = []
                self.domain_indices = {}
                start_idx = 0
                for i, dn in enumerate(domain_name):
                    domain_imgs = ImageFolder(os.path.join(root_dir, dn)).imgs
                    self.imgs += domain_imgs
                    domain_labels_list += [i] * len(domain_imgs)
                    end_idx = start_idx + len(domain_imgs)
                    self.domain_indices[i] = list(range(start_idx, end_idx))
                    start_idx = end_idx
                self.domain_labels = np.array(domain_labels_list)
                self.domain_num = len(domain_name)
            else:
                self.imgs = ImageFolder(os.path.join(root_dir, domain_name)).imgs
                self.domain_labels = np.array([0] * len(self.imgs))
                self.domain_num = 1

            img_paths = [item[0] for item in self.imgs]
            class_labels = [item[1] for item in self.imgs]
            self.labels = np.array(class_labels)
            self.x = img_paths

        self.indices = np.arange(len(self.imgs)) if indices is None else indices

        if mode == 'Default':
            self.loader = default_loader
        elif mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        # --- CORRECTION ---
        # Explicitly create the .samples attribute for external access.
        # This makes the dataset fully compatible with the getdataloader logic.
        self.samples = list(zip(self.x, self.labels.tolist(), self.domain_labels.tolist()))


    def input_trans(self, x):
        """Apply transformation to input image"""
        return self.transform(x) if self.transform is not None else x

    def __getitem__(self, index):
        """Get an item by index"""
        index = self.indices[index]
        img = self.input_trans(self.loader(self.x[index]))
        ctarget = self.labels[index]
        domain_label = self.domain_labels[index]
        return img, ctarget, domain_label

    def __len__(self):
        """Return dataset length"""
        return len(self.indices)


class DGSampler:
    """
    Sampler for domain generalization that ensures each batch contains
    samples from all domains in balanced proportions.
    """
    def __init__(self, dataset, batch_size):
        """
        Initialize domain generalization sampler.
        
        Args:
            dataset: Dataset with domain_indices attribute
            batch_size: Batch size (should be divisible by number of domains)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.domain_indices = dataset.domain_indices
        self.domain_num = dataset.domain_num
        self.samples_per_domain = batch_size // self.domain_num
        self.domain_iterators = {
            domain: iter(np.random.permutation(indices))
            for domain, indices in self.domain_indices.items()
        }
        
    def __iter__(self):
        """Yield batches with balanced domain representation"""
        while True:
            batch_indices = []
            for domain in range(self.domain_num):
                try:
                    domain_batch = [next(self.domain_iterators[domain]) 
                                    for _ in range(self.samples_per_domain)]
                except StopIteration:
                    self.domain_iterators[domain] = iter(
                        np.random.permutation(self.domain_indices[domain])
                    )
                    domain_batch = [next(self.domain_iterators[domain]) 
                                   for _ in range(self.samples_per_domain)]
                batch_indices.extend(domain_batch)
            
            yield batch_indices
            
    def __len__(self):
        """Return the number of batches"""
        return len(self.dataset) // self.batch_size
    
        
def get_DG_dataloader(cfg, test_domain_name=None):

    rate = 0.1
    batch_size = cfg.TEST.BATCH_SIZE
    dataset_name = cfg.DG.DATASET
    task = cfg.DG.TASK
    data_dir = complete_data_dir_path(cfg.DATA_DIR, dataset_name)

    source_domains_name, target_domains_name = get_source_target_names(cfg)
    # --- 领域泛化 (Domain Generalization) 任务 ---
    if "DG" in task:
        # 1. 创建一个临时的 "主" 数据集，仅用于加载所有域的样本元数据
        master_dataset = ImageDataset(
            dataset=dataset_name,
            task='DG',
            root_dir=data_dir,
            domain_name=source_domains_name, # 假设 domains 是一个列表，包含所有域的名称
            transform=image_train(dataset_name), # 变换可以在后续实例化时传入
        )
        
        all_samples = master_dataset.samples
        # 提取用于分层的标签（这里使用域标签）
        stratify_labels = [sample[2] for sample in all_samples] 
        
        train_samples, val_samples = ms.train_test_split(
            all_samples,
            test_size=rate,
            random_state=42,
            stratify=stratify_labels
        )
        
        # 3. 使用划分好的样本，创建独立的、功能完整的 ImageDataset 实例
        train_dataset = ImageDataset(
            dataset=dataset_name,
            task='DG',
            root_dir=data_dir, # root_dir 可能仍需要，用于拼接完整路径
            domain_name=source_domains_name, # 假设 domains 是一个列表，包含所有域的名称
            transform=image_train(dataset_name),
            samples=train_samples,  # 传入预先划分好的训练样本
        )
        
        val_dataset = ImageDataset(
            dataset=dataset_name,
            task='DG',
            root_dir=data_dir,
            domain_name=source_domains_name, # 假设 domains 是一个列表，包含所有域的名称
            transform=image_test(dataset_name),
            samples=val_samples,  # 传入预先划分好的验证样本
        )
        
        print(f"DG Task: Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

        # 4. 创建 DataLoaders
        dg_sampler = DGSampler(train_dataset, batch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=dg_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
            
    # --- 领域自适应 (Domain Adaptation) 或其他任务的逻辑可以类似地修改 ---
    elif "DA" in task:
        # 对于DA，通常有一个源域和一个目标域，划分逻辑可能不同
        # 这里我们假设是对单个域进行标准的训练/验证划分
        master_dataset = ImageDataset(
            dataset=dataset_name,
            task='DA',
            root_dir=data_dir,
            domain_name=source_domains_name, 
            transform=image_train(dataset_name),
        )

        all_samples = master_dataset.samples
        # 根据类别标签进行分层
        stratify_labels = [sample[1] for sample in all_samples]

        train_samples, val_samples = ms.train_test_split(
            all_samples,
            test_size=rate,
            random_state=42,
            stratify=stratify_labels
        )

        train_dataset = ImageDataset(
            dataset=dataset_name, 
            task='DA', 
            root_dir=data_dir, 
            domain_name=source_domains_name, # 假设 domains 是一个列表，包含所有域的名称

            transform=image_train(dataset_name), 
            samples=train_samples,
        )
        
        val_dataset = ImageDataset(
            dataset=dataset_name, 
            task='DA', 
            root_dir=data_dir, 
            domain_name=source_domains_name, # 假设 domains 是一个列表，包含所有域的名称
            transform=image_test(dataset_name),
            samples=val_samples,
        )

        print(f"DA Task: Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
    
    if "test" in task:
        # 测试集通常不需要划分
        test_dataset = ImageDataset(
            dataset=dataset_name,
            task='test', # 任务名也应为 'test'
            root_dir=data_dir,
            domain_name=[test_domain_name] if test_domain_name else target_domains_name, # 如果有特定目标域，则使用它
            transform=image_test(dataset_name), # 使用测试专用的图像变换
        )
        
        print(f"Test Task: Test dataset size: {len(test_dataset)}")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True, # 测试时通常不打乱顺序
            num_workers=4,
            pin_memory=True
        )
        

    return train_loader, val_loader, test_loader if "test" in task else None