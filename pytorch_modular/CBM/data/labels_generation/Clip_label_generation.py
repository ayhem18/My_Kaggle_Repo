"""
This script contains the main class responsible for generating concept-labels for a Concept Bottleneck model
"""
from PIL import Image
import torch
import clip

from torch import nn
from typing import List, Union, Iterable
from pathlib import Path
from pytorch_modular.CBM.data.models import CBM_SingletonInitializer


class ClipLabelGenerator:
    def __init__(self):
        self.clip_model, self.image_processor, self.device = (CBM_SingletonInitializer().get_clip_model(),
                                                              CBM_SingletonInitializer().get_clip_processor(),
                                                              CBM_SingletonInitializer().get_device())
        # make sure to put the clip_model and the image processor to the device 
        self.clip_model.to(self.device)

        if hasattr(self.image_processor, 'to'):
            self.image_processor.to(self.device)

        # create a softmax torch layer to finalize the labels: convert them from cosine differences to probability-like values
        self.softmax_layer = nn.Softmax(dim=1).to(device=self.device)

    def encode_concepts(self, concepts: List[str]) -> torch.Tensor:
        # tokenize the text
        concepts_tokens = clip.tokenize(concepts).to(self.device)

        with torch.no_grad():
            # encode the concepts: features
            concepts_clip = self.clip_model.encode_text(concepts_tokens)

        # make sure the embeddings are of the expected shape
        batch_size, embedding_dim = concepts_clip.shape

        if batch_size != len(concepts):
            raise ValueError((f"Please make sure the batch size of the CLIP text embeddings match the number of concepts\n"
                              f"number of concepts: {len(concepts)}. batch size found: {batch_size}"))
        
        return concepts_clip

    def generate_image_label(self,
                             images: Union[Iterable[Union[str, Path, torch.Tensor]], str, Path, torch.Tensor],
                             concepts_features: Union[torch.Tensor, List[str]],
                             ) -> torch.Tensor:

        # if only one image was passed, wrap it in a list
        if not isinstance(images, Iterable):
            images = [images]

        # if the images are passed as paths, read them
        if isinstance(images[0], (str, Path)):
            images = [Image.open(i) for i in images]

        # process the image: process each image with the CLIP processor (the CLIP.processor does not seem to support batching)
        # convert them to Tensors and stack them into a single tensor
        processed_images = torch.stack([self.image_processor(im).to(self.device) for im in images])

        # proceeding depending on the type of the passed 'concepts'
        if isinstance(concepts_features, List) and isinstance(concepts_features[0], str):
            # if the given concepts are in textual form then we can pass the data directly to the CLIP model
            logits_per_image, _ = self.clip_model(images, concepts_features)
            # as per the documentation of the CLIP model: https://github.com/openai/CLIP
            # logits_per_image represents the cosine difference between the embedding of the images with respect
            # to the given textual data
            return logits_per_image

        # if the data is given as a tensor, then compute the cosine difference
        image_embeddings = self.clip_model.encode_image(processed_images)
        # normalize both the image and concepts embeddings
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        concepts_features /= concepts_features.norm(dim=-1, keepdim=True)

        num_concepts, emb_text_dim = concepts_features.size()
        num_images, emb_img_dim = image_embeddings.size() 

        if emb_text_dim != emb_img_dim:
            raise ValueError((f"In the current setting, image embddings do not match text embddings size-wise.\n"
                              f"Found: text dim: {emb_text_dim}. img dim: {emb_img_dim}"))

        cn = concepts_features.norm(dim=1, dtype=torch.float32)
        imn = image_embeddings.norm(dim=1, dtype=torch.float32)

        # make sure the tensors are normalized
        if not torch.allclose(cn, torch.ones(num_concepts, dtype=torch.float32).to(device=self.device), atol=10**-3):
            raise ValueError(f"The features are not normalized correctly")

        if not torch.allclose(imn, torch.ones(num_images, dtype=torch.float32).to(device=self.device), atol=10**-3):
            raise ValueError(f"The features are not normalized correctly")

        # return the cosine difference between every image, concept tuple
        cosine_diffs = image_embeddings @ concepts_features.T

        # the final step is to pass the cosine differences through the softmax layer. 
        labels = self.softmax_layer(cosine_diffs)

        return labels
