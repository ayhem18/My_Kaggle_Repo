"""
DISCLAIMER: This script is mainly a personalized version of the concepts generation process suggested by the authors of
the 'Label-Free Concept Bottleneck Models' paper.

The original code can be found through this link: "https://github.com/Trustworthy-ML-Lab/Label-free-CBM"
"""

import random
import torch
import clip
import math
import numpy as np

from tenacity import retry, wait_random_exponential, stop_after_attempt, stop_after_delay
from typing import Iterable
from sentence_transformers import SentenceTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_CLIP_MODEL = "ViT-B/16"
PREFIXES = ["", "a ", "an ", "the "]


# the first step in the filtering process
# the first step is filtering lengthy concepts

def filter_too_long(concepts: Iterable[str],
                    max_length: int,
                    print_prob: float = 0.25):
    """
    deletes all concepts longer than max_len
    """
    filtered_concepts = []
    for c in concepts:
        if len(c) <= max_length:
            filtered_concepts.append(c)
        else:
            if random.random() < print_prob:
                print(f'removing {c} because of its length: {len(c)} > {max_length}')
    return filtered_concepts


def _clip_dot_prods(sequence1: Iterable[str],
                    sequence2: Iterable[str],
                    device: str = DEVICE,
                    clip_name=DEFAULT_CLIP_MODEL,
                    batch_size: int = 256):
    """Returns: numpy array with dot products"""

    # first load the clip model
    clip_model, _ = clip.load(clip_name, device=device)
    text1, text2 = clip.tokenize(sequence1).to(device), clip.tokenize(sequence2).to(device)

    features1 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text1)/batch_size)):
            features1.append(clip_model.encode_text(text1[batch_size*i:batch_size*(i+1)]))
        features1 = torch.cat(features1, dim=0)
        features1 /= features1.norm(dim=1, keepdim=True)

    features2 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text2)/batch_size)):
            features2.append(clip_model.encode_text(text2[batch_size*i:batch_size*(i+1)]))
        features2 = torch.cat(features2, dim=0)
        features2 /= features2.norm(dim=1, keepdim=True)

    dot_prods = features1 @ features2.T
    return dot_prods.cpu().numpy()


def filter_too_similar_to_cls(concepts, classes, cls_sim_cutoff, device="cuda", print_prob=0):
    # first check simple text matches
    print(len(concepts))
    concepts = list(concepts)
    concepts = sorted(concepts)

    for cls in classes:
        for prefix in ["", "a ", "A ", "an ", "An ", "the ", "The "]:
            try:
                concepts.remove(prefix + cls)
                if random.random() < print_prob:
                    print("Class:{} - Deleting {}".format(cls, prefix + cls))
            except(ValueError):
                pass
        try:
            concepts.remove(cls.upper())
        except(ValueError):
            pass
        try:
            concepts.remove(cls[0].upper() + cls[1:])
        except(ValueError):
            pass
    print(len(concepts))

    mpnet_model = get_sentence_transformer('all-mpnet-base-v2')

    class_features_m = mpnet_model.encode(classes)
    concept_features_m = mpnet_model.encode(concepts)
    dot_prods_m = class_features_m @ concept_features_m.T
    dot_prods_c = _clip_dot_prods(classes, concepts)
    # weighted since mpnet has higher variance
    dot_prods = (dot_prods_m + 3 * dot_prods_c) / 4

    to_delete = []
    for i in range(len(classes)):
        for j in range(len(concepts)):
            prod = dot_prods[i, j]
            if prod >= cls_sim_cutoff and i != j:
                if j not in to_delete:
                    to_delete.append(j)
                    if random.random() < print_prob:
                        print("Class:{} - Concept:{}, sim:{:.3f} - Deleting {}".format(classes[i], concepts[j],
                                                                                       dot_prods[i, j], concepts[j]))
                        print("".format(concepts[j]))

    to_delete = sorted(to_delete)[::-1]

    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts


def filter_too_similar(concepts, sim_cutoff, device="cuda", print_prob=0):
    mpnet_model = get_sentence_transformer('all-mpnet-base-v2')
    concept_features = mpnet_model.encode(concepts)

    dot_prods_m = concept_features @ concept_features.T
    dot_prods_c = _clip_dot_prods(concepts, concepts)

    dot_prods = (dot_prods_m + 3 * dot_prods_c) / 4

    to_delete = []
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            prod = dot_prods[i, j]
            if prod >= sim_cutoff and i != j:
                if i not in to_delete and j not in to_delete:
                    to_print = random.random() < print_prob
                    # Deletes the concept with lower average similarity to other concepts - idea is to keep more general concepts
                    if np.sum(dot_prods[i]) < np.sum(dot_prods[j]):
                        to_delete.append(i)
                        if to_print:
                            print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j], dot_prods[i, j],
                                                                              concepts[i]))
                    else:
                        to_delete.append(j)
                        if to_print:
                            print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j], dot_prods[i, j],
                                                                              concepts[j]))

    to_delete = sorted(to_delete)[::-1]
    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts


# since the SentenceTransformer is downloading when used for the first time, Network Errors are bound to raise
@retry(wait=wait_random_exponential(multiplier=2, min=2, max=15), stop=(stop_after_delay(20) | stop_after_attempt(10)))
def get_sentence_transformer(model_name: str = 'all-mpnet-base-v2'):
    return SentenceTransformer(model_name)


def filter_concepts(concepts: Iterable[str],
                    classes: Iterable[str],
                    max_length: int,
                    cls_sim_off: float,  # this is similarity cutoff between concept and classes
                    sim_cutoff: float,  # this is similarity cutoff between concepts
                    batch_size: int = 256,
                    print_prob: float = 0.2):
    # the first step is to remove lengthy concepts
    new_concepts = filter_too_long(concepts=concepts, max_length=max_length, print_prob=print_prob)
    # remove concepts that are too similar to the given classes

    print("Filtering concepts with class similarity started !!!")
    new_concepts = filter_too_similar_to_cls(concepts=new_concepts,
                                             classes=classes,
                                             cls_sim_cutoff=cls_sim_off,
                                             print_prob=print_prob)

    # remove concepts that are too similar to each other
    print("Filtering concepts by mutual concept-similarity started !!!")
    new_concepts = filter_too_similar(concepts=new_concepts,
                                      sim_cutoff=sim_cutoff,
                                      print_prob=print_prob)

    return new_concepts
