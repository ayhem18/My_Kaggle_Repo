# load the data from pytorch
import torch
import os
from torchvision import datasets, transforms as tr
import numpy as np
from collections import Counter
from tqdm import tqdm


def load_data():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_dir, 'data')
    os.makedirs(data_folder, exist_ok=True)

    # apply toTensor transform to each of the images
    basic_transform  = tr.Compose([tr.ToTensor()])
    # use Pytorch to load the dataset
    cifar_train = datasets.CIFAR10(root=os.path.join(data_folder, 'train'), train=True, download=True, transform=basic_transform)
    cifar_test = datasets.CIFAR10(root=os.path.join(data_folder, 'test'), train=False, download=True, transform=basic_transform)

    # build a loader for each split
    from torch.utils.data import DataLoader
    # convert the dataset to a Dataloader for easier manipulation
    train_loader = DataLoader(cifar_train, batch_size=1000, shuffle=False) 
    test_loader = DataLoader(cifar_test, batch_size=1000, shuffle=False)


    # save each of test and train splits in a single tensor
    # the after stacking the different batches together: the output will be of the shape: 50, 1000, 3, 32, 32
    train_tensor = torch.stack([data for data, _ in train_loader])
    test_tensor = torch.stack([data for data, _ in test_loader])


    # converting the different batches into a single large one while flattening each image in the batch 
    # in other words, we will have a large tensor of shape (number of samples, number of pixels )
    train_np = train_tensor.permute((0, 1, 3, 4, 2)).reshape(shape=(len(cifar_train), -1)).numpy()
    test_np = test_tensor.permute((0, 1, 3, 4, 2)).reshape(shape=(len(cifar_test), -1)).numpy()
        
    train_labels = torch.stack([labels for _, labels in train_loader]).reshape((-1,)).numpy()
    test_labels = torch.stack([labels for _, labels in test_loader]).reshape((-1,)).numpy()

    # make sure the shapes are as expected
    print(train_np.shape)
    print(test_np.shape)
    print(train_labels.shape)
    print(test_labels.shape)

    return train_np, train_labels, test_np, test_labels


class KnnClassifier:
    def __init__(self, k: int = 1) -> None:
        self.k = k
        self._train_data = None
        self._train_labels = None           

    def fit(self, train_data: np.ndarray, labels: np.ndarray):
        # make sure to convert the data to numpy arrays
        # saving the data as np.unint8 speeds the algorithm
        self._train_data = train_data.astype(np.uint8)
        self._train_labels = labels.astype(np.uint8)

    def _predict(self, image: np.ndarray) -> int:        
        # first apply np.abs(image - self._train_data): numpy will broadcast the image from (1, num_pixels) to (50000, num_pixes) 
        # np.abs(image-self._Train_data) will be the difference in absolute value between the image and each item in the training data
        # np sum() sums these difference and sum_difference is a numpy array of the shape (50000, 1)
        sum_difference = np.sum(np.abs(image - self._train_data), axis=-1)
        # find the indices of the 'k' nearest training samples: which correspond to the ones with the least difference        
        k_indices = np.argsort(sum_difference, axis=-1)[:self.k].squeeze()
        # use the extracted indices to extract the labels
        k_labels = self._train_labels[k_indices] 
        # use majority voting: Counter counts the frequency of elements and most_common()functions returns a sorted list: [(item, frequency)]
        # indexing by [0][0] returns the most common label in the neighborhood
        return Counter(k_labels).most_common()[0][0]
    
    def predict(self, test_data:np.ndarray):
        # apply list comprehension of efficency
        return [self._predict(t) for t in tqdm(test_data)]

if __name__ == '__main__':
    train_np, train_labels, test_np, test_labels = load_data()
    # initialize a Knn classifier with k=5
    classifier = KnnClassifier(5)
    classifier.fit(train_np, train_labels)
    predictions = classifier.predict(test_np)

    accuray = np.mean((predictions == test_labels))
    print(round(accuray, 3))