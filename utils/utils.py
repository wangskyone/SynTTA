import os
import sys
import torch
import logging
import numpy as np
sys.path.append(f"{os.getcwd()}/datasets")
from imagenet_subsets import IMAGENET_D_MAPPING
from scipy.signal import savgol_filter

from copy import deepcopy
import matplotlib.pyplot as plt
from conf import get_DG_domain_fullnames

logger = logging.getLogger(__name__)

def split_results_by_domain(domain_dict, data, predictions):
    """
    Separate the labels and predictions by domain
    :param domain_dict: dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
    :param data: list containing [images, labels, domains, ...]
    :param predictions: tensor containing the predictions of the model
    :return: updated result dict
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict, domain_seq=None):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    :param domain_dict: dictionary containing the labels and predictions for each domain
    :param domain_seq: if specified and the domains are contained in the domain dict, the results will be printed in this order
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    dom_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting up the results by domain...")
    for key in dom_names:
        content = np.array(domain_dict[key])
        correct.append((content[:, 0] == content[:, 1]).sum())
        num_samples.append(content.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 device: torch.device = None,
                 ):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    correct = 0.
    domain_acc_list = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            predictions = output.argmax(1)

            correct += (predictions == labels.to(device)).float().sum()
            domain_acc_list.append((predictions == labels.to(device)).float().mean().item())

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

    accuracy = correct.item() / len(data_loader.dataset)
    return accuracy, domain_dict, domain_acc_list  



def draw_max_probs(max_probs, arch):
    # Generate synthetic data
    batch_ids = np.arange(len(max_probs))
                   
    avg = [np.mean(max_probs[i]) for i in range(len(max_probs))]

    savgol_smoothed = savgol_filter(avg, 51, 3)

    np.save(f'logitnorm_avg_{arch}.npy', savgol_smoothed)
    # np.save(f'overcf_avg_{arch}.npy', avg)
    # Plot the data with shading
    plt.figure(figsize=(8, 6))
    plt.plot(batch_ids, avg, color='r', label='Max Softmax Probability', linewidth=2, alpha=0.4)
    plt.plot(batch_ids, savgol_smoothed, color='b', label='Savitzky-Golay Smoothed', linewidth=2, alpha=0.8)


    # Adding labels and grid
    plt.xlabel('Batch ID', fontsize=12)
    plt.ylabel('Max Softmax Probability', fontsize=12)
    plt.grid(True)
    plt.title(f'Max Softmax Probability with Confidence Interval on {arch}', fontsize=14)

    # Display the plot
    plt.show()
    # plt.savefig(f'overcf_plot_{arch}.png')
    plt.savefig(f'logitnorm_plot_{arch}.png')


def get_source_target_names(cfg):
    """
    Get the source and target domain names from the configuration.
    :param cfg: configuration object
    :return: source domain name, target domain names
    """
    dataset_name = cfg.DG.DATASET
    domains = get_DG_domain_fullnames(dataset_name)
    source_domains_name = [domains[i] for i in cfg.DG.TRAINING_DOMAINS]
    target_domains_name = [domains[i] for i in cfg.DG.TESTING_DOMAINS]
    
    return source_domains_name, target_domains_name


def draw_drift(drift_history, domain_names, method_name, dataset_name):
    """
    绘制模型在多个连续领域上的参数漂移曲线。

    Args:
        drift_history (list[list[float]]): 一个嵌套列表。
            其长度为 n，对应 n 个领域。
            drift_history[i] 是一个列表，包含第 i 个领域中，
            每个批次（batch）的参数漂移值。
        domain_names (list[str]): 一个包含 n 个领域名称的列表。
    """
    if len(drift_history) != len(domain_names):
        raise ValueError("drift_history 和 domain_names 的长度必须相等。")
    
    # save drift history for further analysis
    np.save(f'bn_drift_{method_name}_{dataset_name}.npy', np.array(drift_history))


    # --- 设置画布 ---
    plt.style.use('seaborn-v0_8-whitegrid') # 使用更美观的绘图风格
    fig, ax = plt.subplots(figsize=(18, 8)) # 使用更大的画布以容纳所有领域

    # --- 绘制数据 ---
    total_batch_counter = 0
    # 为每个领域分配一个不同的颜色
    colors = plt.cm.get_cmap('viridis', len(domain_names))

    for i, domain_drift_values in enumerate(drift_history):
        num_batches_in_domain = len(domain_drift_values)
        
        # 创建当前领域的 x 轴坐标（全局批次索引）
        x_values = np.arange(total_batch_counter, total_batch_counter + num_batches_in_domain)
        
        # 绘制当前领域的漂移曲线
        ax.plot(x_values, domain_drift_values, color=colors(i), linewidth=2)
        
        # 在每个领域的交界处绘制一条垂直虚线
        if i > 0:
            ax.axvline(x=total_batch_counter, color='grey', linestyle='--', linewidth=1.2)
            
        # 在每个领域的中间顶部添加领域名称标注
        domain_center_x = total_batch_counter + num_batches_in_domain / 2
        # 将文本放在图表顶部，避免遮挡曲线
        ax.text(domain_center_x, ax.get_ylim()[1] * 1.05, domain_names[i], 
                ha='center', va='bottom', fontsize=12, weight='bold')

        # 更新全局批次计数器
        total_batch_counter += num_batches_in_domain

    # --- 美化图表 ---
    ax.set_title('BN Parameter Drift Across Sequential Domains', fontsize=18, weight='bold')
    ax.set_xlabel('Global Batch Index (Time)', fontsize=14)
    ax.set_ylabel('Average L2 Distance from Source Parameters', fontsize=14)
    
    # 调整y轴的范围，给顶部的文本留出空间
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 调整布局以防止标签重叠
    plt.tight_layout()
    plt.show()
    plt.savefig(f'bn_drift_{method_name}_{dataset_name}.png', bbox_inches='tight')