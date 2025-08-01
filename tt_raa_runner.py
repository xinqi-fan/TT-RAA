import random
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *
from utils_kl import *
from datetime import datetime


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings on specific dataset in yaml format.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'RN101', 'ViT-B/16', 'ViT-B/32'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args

def update_cache(cache, pred, features_loss, shot_capacity):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]

def update_cache_gmm(cache, pred, features_loss, eta):
    """Update cache with new features and loss"""
    num_features = features_loss[0].shape[1]
    device = features_loss[0].device
    std_dev = 0.1
    eta_cov = eta
    with torch.no_grad():
        item = features_loss
        if pred in cache:
            if features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1][0] = (1-eta) * cache[pred][-1][0] + eta * item[0]
                cache[pred][-1][1] = (1-eta) * cache[pred][-1][1] + eta * item[1]
                covariance_matrix_new = torch.mm(item[0] - cache[pred][-1][0], (item[0] - cache[pred][-1][0]).t())
                cache[pred][-1][2] = (1-eta_cov) * cache[pred][-1][2] + eta_cov * covariance_matrix_new
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            # key: label, values: [mean, entropy, covariane]
            cache[pred] = [item] 
            covariance_matrix = torch.eye(num_features) * (std_dev ** 2)
            cache[pred][-1].append(covariance_matrix.to(device))

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights):
    """Compute logits using vision space similarity retrieval"""
    with torch.no_grad():
        cache_keys = [] # G (mean)
        cache_values = [] # L
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits
    
def compute_cache_logits_with_cov(image_features, cache, alpha, clip_weights):
    """Compute logits using Gaussian discriminant."""
    with torch.no_grad():
        cache_keys = [] # G
        cache_values = [] # L
        cache_covs = [] # Sigma
        num_classes = clip_weights.size(1)
        score = torch.zeros((1, num_classes), dtype=image_features.dtype, device=image_features.device)
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                cache_values.append(class_index)
                cache_covs.append(item[2])

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_covs = torch.stack(cache_covs)

        # matrix form
        mean = cache_keys.permute(1, 0)  # mean # current_num_class*512  
        cov = torch.mean(cache_covs, dim=0) # Mean of covariacne matrixes for all current classes
        cov_inv = torch.pinverse(cov).half()
        score = mean @ cov_inv @ image_features.t() - 0.5 * torch.diag(mean @ cov_inv @ mean.t()).unsqueeze(-1) + torch.log(torch.tensor(1/num_classes, dtype=image_features.dtype, device=image_features.device))

        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half() # L

        new_knowledge = image_features @ cache_keys
        score_scaled = scale_((score).cuda(), new_knowledge) / 3.0 
        logits = score_scaled.permute(1,0) @ cache_values
        return alpha * logits

def compute_KL_logits(image_features, cache, alpha, clip_weights, temperature=0.2):
    """Compute KL logits."""
    with torch.no_grad():
        cache_keys = [] # G
        cache_values = [] # L
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        test_kl_divs_sims = get_kl_div_sims(image_features, cache_keys, clip_weights, temperature)
        
        new_knowledge = image_features @ cache_keys
        neg_affs = scale_((test_kl_divs_sims).cuda(), new_knowledge)
        kl_logits = -neg_affs.half() @ cache_values
        return alpha * kl_logits


def run_test_ttraa(cfg, loader, clip_model, clip_weights):
    with torch.no_grad():
        database, accuracies = {}, []

        #Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, _, pred = get_clip_logits(images ,clip_model, clip_weights)
            target = target.cuda()

            # update streaming mixture of Gaussian database 
            update_cache_gmm(database, pred, [image_features, loss], cfg['eta'])

            # compute logits
            final_logits = clip_logits.clone()
            final_logits += compute_cache_logits(image_features, database, cfg['alpha'], cfg['beta'], clip_weights)
            final_logits += compute_cache_logits_with_cov(image_features, database, cfg['alpha'], clip_weights)
            final_logits += compute_KL_logits(image_features, database, cfg['alpha'], clip_weights)
                
            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)

            if i%1000==0:
                print("---- test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        print("---- test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
        return sum(accuracies)/len(accuracies)



def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    
    # Run TT-RAA on each dataset
    dataset_names = []
    accs = []
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        acc = run_test_ttraa(cfg, test_loader, clip_model, clip_weights)

        dataset_names.append(dataset_name)
        accs.append(acc)
    print(dataset_names)
    print(accs)

    # save logs
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_results = np.row_stack((dataset_names, accs))
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join("results", f"{timestamp}.csv")
    np.savetxt(log_path, log_results, delimiter=",", fmt="%s", comments="")

if __name__ == "__main__":
    main()