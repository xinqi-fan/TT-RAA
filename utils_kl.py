import torch
import torch.nn.functional as F

def scale_(x, target):
    
    y = (x - x.min()) / ((x.max() - x.min()) + 1e-8)
    y *= target.max() - target.min()
    y += target.min()
    
    return y

def compute_image_text_distributions(temp, train_images_features_agg, test_features, vanilla_zeroshot_weights):
    train_image_class_distribution = train_images_features_agg.T @ vanilla_zeroshot_weights # Psi
    train_image_class_distribution = torch.nn.Softmax(dim=-1)(train_image_class_distribution/temp)

    test_image_class_distribution = test_features @ vanilla_zeroshot_weights # psi
    test_image_class_distribution = torch.nn.Softmax(dim=-1)(test_image_class_distribution/temp)
    
    return train_image_class_distribution, test_image_class_distribution

def get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution, eps=1e-10):
    # Clamp to avoid log(0)
    train = train_image_class_distribution.clamp(min=eps)
    test = test_image_class_distribution.clamp(min=eps)

    # Reshape for broadcasting: test [M, 1, K], train [1, N, K]
    test = test.unsqueeze(1)  # [M, 1, K]
    train = train.unsqueeze(0)  # [1, N, K]

    # KL divergence: sum_i test_i * (log(test_i) - log(train_i))
    kl_div = test * (test.log() - train.log())  # [M, N, K]
    kl_div = kl_div.sum(dim=-1)  # [M, N]

    return kl_div

def get_kl_div_sims(test_features, train_features, clip_weights, temperature=0.2):
    train_image_class_distribution, test_image_class_distribution = compute_image_text_distributions(temperature, train_features, test_features, clip_weights)

    test_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution)

    return test_kl_divs_sim