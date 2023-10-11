import os, sys
from pathlib import Path

HOME = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(HOME, 'data')

current = HOME



while 'pytorch_modular' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'pytorch_modular'))


from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tr
from building_blocks import UnderCompleteAE
from pytorch_modular.image_classification import engine_classification as engine
from torch.optim import Adam 
from torch.optim.lr_scheduler import LinearLR
from torch.nn import MSELoss


# since the task is not classification, a new dataset should be created to fit our training objectives
class GenerativeWrapper(Dataset):
    def __init__(self, data: Dataset) -> None:
        super().__init__()
        self.original_data = data

    def __getitem__(self, index):
        data, _ = self.original_data.__getitem__(index)
        return (data, data)
    
    def __len__(self):
        return len(self.original_data)
    


def solution():
    mnist_train = MNIST(root=DATA_FOLDER, train=True, download=True, transform=tr.Compose([tr.ToTensor(), lambda x: x.reshape(-1)]))
    mnist_test = MNIST(root=DATA_FOLDER, train=False, download=True, transform=tr.Compose([tr.ToTensor(), lambda x: x.reshape(-1)]))

    from sklearn.model_selection import train_test_split
    # create the val split
    mnist_train, mnist_val = train_test_split(mnist_train, random_state=69, test_size=0.1)

    # wrap the different splits
    mnist_train = GenerativeWrapper(mnist_train)
    mnist_val = GenerativeWrapper(mnist_val)
    mnist_test = GenerativeWrapper(mnist_test)


    # create dataloader
    train_dl = DataLoader(dataset=mnist_train, 
                        batch_size=512, 
                        shuffle=True,
                        pin_memory=True,
                        drop_last=True,
                        num_workers=os.cpu_count() // 2, 
                        ) 

    val_dl = DataLoader(dataset=mnist_val, 
                        batch_size=512, 
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        num_workers=os.cpu_count() // 2) 

    for im, label in train_dl:
        print(im.shape)
        break

    model = UnderCompleteAE(in_features=784, bottleneck=256, num_layers=2)

    # training     
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = LinearLR(optimizer=optimizer, start_factor=1, end_factor=0.05, total_iters=100)

    output_layer = lambda x: x
    loss_function = MSELoss()

    train_configuration = {"optimizer": optimizer, 
                        'scheduler': scheduler,
                        'output_layer': output_layer,
                        'loss_function': loss_function,
                        'min_val_loss': 10 ** -4,
                        'max_epochs': 25,
                        'report_epoch': 5,
                        'metrics': {}
                        }
    
    engine.train_model(model=model, 
                    train_dataloader=train_dl, 
                    test_dataloader=val_dl, 
                    train_configuration=train_configuration,
                    log_dir=os.path.join(HOME, 'runs'),
                    )


if __name__ == '__main__':
    solution()

