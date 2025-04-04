import os
import sys
import warnings
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

import torch
from torchvision import transforms
from torch.utils.data import IterableDataset

from captum.concept import TCAV, Concept
from captum.attr import LayerIntegratedGradients
from captum.concept._utils.data_iterator import dataset_to_dataloader

# local import for creating the resnet18 model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model import create_resnet18

# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None`.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Using default classifier for TCAV.*")

# device setup
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


# transform function
def transform(img):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(img)


# simple iterable dataset for feeding concepts
class SimpleIterableDataset(IterableDataset):
    def __init__(self, data_loader_fn):
        self.data_loader_fn = data_loader_fn
    def __iter__(self):
        return iter(self.data_loader_fn())


# data loading helper functions
def load_image_tensors(attr_name, attr_file_path='../data/list_attr_celeba.txt',
                       image_dir='../data/img_align_celeba/', transform=transform,
                       max_images=100, num_random_set=2):
    """
    Load image tensors for a given attribute with a specified number of positive images and 
    negative images per random set.
    """
    df = pd.read_csv(attr_file_path, sep='\s+', skiprows=1)
    df.index.name = 'filename'

    if attr_name not in df.columns:
        raise ValueError(f"Attribute: '{attr_name}' not found in {attr_file_path}")

    pos_df = df[df[attr_name] == 1]
    neg_df = df[df[attr_name] == -1]

    if len(pos_df) < max_images or len(neg_df) < (max_images * num_random_set):
        raise ValueError(f"Not enough samples for '{attr_name}'.")

    pos_sample = pos_df.sample(n=max_images, random_state=42)
    neg_sets = [neg_df.sample(n=max_images, random_state=42 + i) for i in range(num_random_set)]

    def load_images(filenames):
        return [transform(Image.open(os.path.join(image_dir, f)).convert('RGB')) for f in filenames]

    pos_images = load_images(pos_sample.index.tolist())
    neg_image_sets = [load_images(neg.index.tolist()) for neg in neg_sets]

    return pos_images, neg_image_sets


def assemble_concept(concept_name, concept_id, image_tensors):
    """
    Create a Captum Concept object from a list of image tensors.
    """
    def loader():
        for img in image_tensors:
            yield img.to(DEVICE).float()

    dataset = SimpleIterableDataset(loader)
    data_iter = dataset_to_dataloader(dataset)
    return Concept(id=concept_id, name=concept_name, data_iter=data_iter)


def assemble_all_concepts(exclude_attr, attr_file_path='../data/list_attr_celeba.txt',
                          image_dir='../data/img_align_celeba/', transform=transform,
                          max_images=100, num_random_set=2, starting_concept_id=0):
    """
    Assemble a list of all concepts (and their associated random sets) except one that is excluded.
    """
    df = pd.read_csv(attr_file_path, sep='\s+', skiprows=1)
    all_attrs = df.columns.tolist()

    if exclude_attr not in all_attrs:
        raise ValueError(f"Attribute '{exclude_attr}' not found in CelebA attributes.")

    concepts = []
    concept_id = starting_concept_id

    for attr in all_attrs:
        if attr == exclude_attr:
            continue
        try:
            pos_imgs, neg_sets = load_image_tensors(attr, attr_file_path, image_dir, transform, max_images, num_random_set)
            # Add the positive concept
            concepts.append(assemble_concept(attr, concept_id, pos_imgs))
            concept_id += 1
            # Add random negative concepts for each set
            for i, neg_imgs in enumerate(neg_sets):
                concepts.append(assemble_concept(f'{attr}_random_{i}', concept_id, neg_imgs))
                concept_id += 1
        except Exception as e:
            print(f"Skipping attribute '{attr}' due to error: {e}")
    return concepts


def run_tcav(target_attr="Young", model_path=None, target_class=0,
             attr_file_path='../data/list_attr_celeba.txt',
             image_dir='../data/img_align_celeba/', max_images=100, num_random_set=2,
             n_steps=10, processes=0, save_plots=True):
    """
    Run the TCAV experiment for a specified target attribute and model.
    
    Parameters:
        target_attr (str): The attribute used as the target for explanation (e.g. "Young").
        model_path (str or None): Path to a saved model. If None, the default pretrained model is used.
        target_class (int): The target class index for which the explanations are generated.
        attr_file_path (str): File path for the CelebA attribute list.
        image_dir (str): Directory where CelebA images are stored.
        max_images (int): Number of images to sample for each concept.
        num_random_set (int): Number of random sets per concept.
        n_steps (int): Number of steps for TCAV.
        processes (int): Number of processes to use (0 means no multiprocessing).
        save_plots (bool): Whether to save the generated bar plots.
        
    Returns:
        df (pd.DataFrame): A dataframe containing the TCAV scores.
        tcav_scores (dict): The raw TCAV score output.
    """
    # load model
    model = create_resnet18(1, pretrained=True)
    model = model.to(DEVICE)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # specify layers for which to compute TCAV scores
    layers = {
        'layer1': model.layer1,
        'layer2': model.layer2,
        'layer3': model.layer3,
        'layer4': model.layer4
    }
    lig = LayerIntegratedGradients(model, None, multiply_by_inputs=False)
    mytcav = TCAV(model=model, layers=layers, layer_attr_method=lig)
    
    # assemble all concepts excluding the target attribute (so that TCAV compares each concept with the target)
    concepts = assemble_all_concepts(exclude_attr=target_attr, attr_file_path=attr_file_path,
                                     image_dir=image_dir, transform=transform,
                                     max_images=max_images, num_random_set=num_random_set)

    # create experimental sets (each concept is compared with its corresponding random sets)
    experimental_sets = []
    pair_name_map = {}
    # we assume that concepts are organized as positive concept and then its random sets in order.
    for i in range(0, len(concepts), 1 + num_random_set):
        real = concepts[i]
        for j in range(1, num_random_set + 1):
            experimental_sets.append([real, concepts[i + j]])
            pair_name_map[f'{i}-{i+j}'] = real.name

    # load target attribute images for which explanations will be generated
    target_images, _ = load_image_tensors(attr_name=target_attr, attr_file_path=attr_file_path,
                                            image_dir=image_dir, transform=transform,
                                            max_images=25, num_random_set=0)
    inputs = torch.stack(target_images).float().to(DEVICE)
    
    # run TCAV
    tcav_scores = mytcav.interpret(inputs=inputs, experimental_sets=experimental_sets,
                                   target=target_class, n_steps=n_steps, processes=processes)

    # define and create output directory
    output_dir = os.path.join('logs', f'{target_attr.lower()}_tcav')
    os.makedirs(output_dir, exist_ok=True)
    
    # display results
    summary_file = os.path.join(output_dir, f'{target_attr.lower()}_tcav_scores.txt')
    with open(summary_file, 'w') as f:
        f.write("TCAV scores summary:\n\n")
        for concept_pair, layer_dict in tcav_scores.items():
            concept_name = pair_name_map.get(concept_pair, concept_pair)
            f.write(f"Attribute: {concept_name} (Pair ID: {concept_pair})\n")
            for layer_name, scores in layer_dict.items():
                sign_count = scores.get('sign_count')
                magnitude = scores.get('magnitude')
                if sign_count is not None and magnitude is not None:
                    tcav_score = float(sign_count[0])
                    f.write(f"   Layer: {layer_name} | TCAV Score: {tcav_score:.3f}\n")
                else:
                    f.write(f"   Layer: {layer_name} | Missing score data\n")
            f.write("\n")


    # create a dataframe from the results for further analysis
    data = []
    for pair_id, layer_dict in tcav_scores.items():
        attribute = pair_name_map.get(pair_id, pair_id)
        for layer_name, scores in layer_dict.items():
            sign_count = scores.get('sign_count')
            if sign_count is not None:
                tcav_score = float(sign_count[0])
                data.append({
                    'attribute': attribute,
                    'layer': layer_name,
                    'pair_id': pair_id,
                    'tcav_score': tcav_score
                })
    df = pd.DataFrame(data)

    # plot TCAV scores for each layer
    layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    for l in layer_names:
        df_layer = df[df['layer'] == l]
        df_mean = df_layer.groupby('attribute', as_index=False)['tcav_score'].mean().sort_values('tcav_score', ascending=False)

        plt.figure(figsize=(12, 6))
        plt.bar(df_mean['attribute'], df_mean['tcav_score'])
        plt.xticks(rotation=90)
        plt.xlabel('Concept (Attribute)')
        plt.ylabel('TCAV Score')
        plt.title(f"TCAV Scores for class '{target_class}' at {l}")
        plt.tight_layout()

        if save_plots:
            plot_path = os.path.join(output_dir, f'{l}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    return df, tcav_scores


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run TCAV experiment for CelebA and ResNet18.")
    parser.add_argument('--target_attr', type=str, default="Young", help="Attribute to be explained.")
    parser.add_argument('--model_path', type=str, default=None, help="Path to saved model (if any).")
    parser.add_argument('--target_class', type=int, default=0, help="Target class index for TCAV.")
    parser.add_argument('--attr_file_path', type=str, default='../data/list_attr_celeba.txt', help="Path to CelebA attributes file.")
    parser.add_argument('--image_dir', type=str, default='../data/img_align_celeba/', help="Directory for CelebA images.")
    parser.add_argument('--max_images', type=int, default=100, help="Maximum images per concept.")
    parser.add_argument('--num_random_set', type=int, default=2, help="Number of random sets per concept.")
    parser.add_argument('--n_steps', type=int, default=10, help="Number of steps for TCAV.")
    parser.add_argument('--processes', type=int, default=0, help="Number of processes to use for TCAV.")
    parser.add_argument('--save_plots', type=bool, default=True, help="Whether to save the plots.")

    args = parser.parse_args()

    run_tcav(target_attr=args.target_attr, model_path=args.model_path, target_class=args.target_class,
             attr_file_path=args.attr_file_path, image_dir=args.image_dir, max_images=args.max_images,
             num_random_set=args.num_random_set, n_steps=args.n_steps, processes=args.processes, save_plots=args.save_plots)
