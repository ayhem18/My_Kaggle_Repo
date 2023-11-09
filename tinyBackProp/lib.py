from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import trange

def load_mnist():
    # Loading MNIST Dataset from official source ubyte files
    
    with open('./data/train-images-idx3-ubyte', 'rb') as f:
        X_train = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
    with open('./data/train-labels-idx1-ubyte', 'rb') as f:
        y_train = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with open('./data/t10k-images-idx3-ubyte', 'rb') as f:
        X_test = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
    with open('./data/t10k-labels-idx1-ubyte', 'rb') as f:
        y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        
    return X_train, y_train, X_test, y_test 
    

def softmax(X: np.ndarray, dim: int = None, dL: np.ndarray = None) -> np.ndarray:
    """Compute softmax values over specified axis or backpropagates the loss.
    
    \\text{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} 
    
    Args:
        X (np.ndarray): input array
        dim (int, optional): axis to compute softmax over. Defaults to -1.
        dL (np.ndarray, optional): backpropagated loss. Defaults to None.
        
    Returns:
        np.ndarray: softmax values or backpropagated loss
    """
    if dL is not None:
        # compute backpropagated loss
        if dim:
            # compute softmax
            e = np.exp(X - np.max(X, axis=dim, keepdims=True))
            s = e / np.sum(e, axis=dim, keepdims=True)
            return np.sum(dL * s, axis=dim, keepdims=True) - s * np.sum(dL * s, axis=dim, keepdims=True)
        else:
            e = np.exp(X - np.max(X))
            s = e / np.sum(e)
            return np.sum(dL * s) - s * np.sum(dL * s)
    else:
        if dim:
        # compute softmax
            e = np.exp(X - np.max(X, axis=dim, keepdims=True))
            return e / np.sum(e, axis=dim, keepdims=True)
        else:
            return np.exp(X - np.max(X)) / np.sum(np.exp(X - np.max(X)))
        
        
def add(x: np.ndarray, b: np.ndarray, dL: np.ndarray = None) -> np.ndarray:
    """Compute addition or backpropagates the loss.
    
    \\text{add}(x, b) = x + b
    
    Args:
        x (np.ndarray): input array
        b (np.ndarray): bias array
        dL (np.ndarray, optional): backpropagated loss. Defaults to None.
        
    Returns:
        np.ndarray: addition or backpropagated loss
    """
    if dL is not None:
        # compute backpropagated loss
        # of b + x w.r.t. b and x
        return dL, dL
    else:
        # compute addition
        return x + b
    

def dot(b: np.ndarray, x: np.ndarray, dL: np.ndarray = None) -> np.ndarray:
    """Compute dot product or backpropagates the loss.
    
    \\text{dot}(x, b) = x \cdot b
    
    Args:
        x (np.ndarray): input array
        b (np.ndarray): bias array
        dL (np.ndarray, optional): backpropagated loss. Defaults to None.
        
    Returns:
        np.ndarray: dot product or backpropagated loss
    """
    if dL is not None:
        # Compute backpropagated loss
        # of bx where b is a matrix
        # and x is a batch of vectors
        
        # Also, return two gradients:
        # 1. gradient w.r.t. b
        # 2. gradient w.r.t. x
        return np.dot(dL, x.T), np.dot(b.T, dL)
    else:
        # compute dot product
        return np.dot(b, x)
    
    
def batch_dot(x: np.ndarray, b: np.ndarray, dL: np.ndarray = None) -> np.ndarray:
    """Compute batch dot product or backpropagates the loss.
    
    \\text{batch_dot}(x, b) = [x_1 \cdot b_1, x_2 \cdot b_2, \dots, x_n \cdot b_n]
    
    Args:
        x (np.ndarray): input array
        b (np.ndarray): bias array
        dL (np.ndarray, optional): backpropagated loss. Defaults to None.
        
    Returns:
        np.ndarray: batch dot product or backpropagated loss
    """
    if dL is not None:
        # compute backpropagated loss
        return dL * b
    else:
        # compute batch dot product
        return np.sum(x * b, axis=1)
    
    
    
def relu(X: np.ndarray, dL: np.ndarray = None) -> np.ndarray:
    """Compute ReLU values or backpropagates the loss.
    
    \\text{ReLU}(x) = \max(0, x)
    
    Args:
        X (np.ndarray): input array
        dL (np.ndarray, optional): backpropagated loss. Defaults to None.
        
    Returns:
        np.ndarray: ReLU values or backpropagated loss
    """
    if dL is not None:
        # compute backpropagated loss
        return dL * (X > 0)
    else:
        # compute ReLU
        return np.maximum(0, X)
    
    
def matmul(X: np.ndarray, W: np.ndarray, dL: np.ndarray = None) -> np.ndarray:
    """Compute matrix multiplication or backpropagates the loss.
    
    \\text{matmul}(X, W) = WX
    
    Args:
        X (np.ndarray): input array
        W (np.ndarray): weight array
        dL (np.ndarray, optional): backpropagated loss. Defaults to None.
        
    Returns:
        np.ndarray: matrix multiplication or backpropagated loss
    """
    if dL is not None:
        # compute backpropagated loss
        return np.dot(dL, W.T), np.dot(X.T, dL)
    else:
        # compute matrix multiplication
        return np.dot(X, W)
    
    

def mse(X: np.ndarray, y: np.ndarray, dL: np.ndarray = None) -> np.ndarray:
    """Compute mean squared error or backpropagates the loss.
    
    \\text{mse}(X, y) = \\frac{1}{2n} \sum_i^n (X_i - y_i)^2
    
    Args:
        X (np.ndarray): input array
        y (np.ndarray): target array
        dL (np.ndarray, optional): backpropagated loss. Defaults to None.
        
    Returns:
        np.ndarray: mean squared error or backpropagated loss
    """
    if dL is not None:
        # compute backpropagated loss
        return (X - y) / X.shape[0]
    else:
        # compute mean squared error
        return np.mean(np.square(np.subtract(X, y))) / 2
    
    
def bce(X: np.ndarray, y: np.ndarray, dL: np.ndarray = None) -> np.ndarray:
    """Compute binary cross entropy or backpropagates the loss.
    
    \\text{bce}(X, y) = -y \log(X) - (1 - y) \log(1 - X)
    
    Args:
        X (np.ndarray): input array
        y (np.ndarray): target array
        dL (np.ndarray, optional): backpropagated loss. Defaults to None.
        
    Returns:
        np.ndarray: binary cross entropy or backpropagated loss
    """
    if dL is not None:
        # compute backpropagated loss
        return -y / X + (1 - y) / (1 - X)
    else:
        # compute binary cross entropy
        return -np.mean(y * np.log(X) + (1 - y) * np.log(1 - X))

    
    
def conv2d(
    X: np.ndarray, 
    k: np.ndarray, 
    dL: np.ndarray = None
) -> np.ndarray:
    """Compute 2D convolution or backpropagates the loss.
    
    conv2d(X, k) = X * k
    
    Args:
        X (np.ndarray): input image of shape (B, C_in, H, W)
        k (np.ndarray): kernel of shape (C_out, C_in, H, W)
        dL (np.ndarray, optional): backpropagated loss. Defaults to None.
        
    Returns:
        np.ndarray: 2D convolution or backpropagated loss
    """
    if dL is not None:
        # Compute backpropagated loss over kernel and input image
        dLdX = np.zeros_like(X)
        dLdK = np.zeros_like(k)

        for i in range(X.shape[0]):
            for j in range(k.shape[0]):
                for h in range(X.shape[2] - k.shape[2] + 1):
                    for w in range(X.shape[3] - k.shape[3] + 1):
                        dLdX[i, :, h:h + k.shape[2], w:w + k.shape[3]] += dL[i, j, h, w] * k[j, :, :, :]
                        dLdK[j, :, :, :] += dL[i, j, h, w] * X[i, :, h:h + k.shape[2], w:w + k.shape[3]]
        
        return dLdX, dLdK

    else:
        output = np.zeros(
            (X.shape[0], k.shape[0], X.shape[2] - k.shape[2] + 1, X.shape[3] - k.shape[3] + 1)
        )
        for i in range(X.shape[0]):
            for j in range(k.shape[0]):
                for h in range(X.shape[2] - k.shape[2] + 1):
                    for w in range(X.shape[3] - k.shape[3] + 1):
                        output[i, j, h, w] = np.sum(X[i, :, h:h + k.shape[2], w:w + k.shape[3]] * k[j, :, :, :])        
        return output
    
def flatten(X: np.ndarray, dL: np.ndarray = None) -> np.ndarray:
    """Compute flattening or backpropagates the loss.
    
    \\text{flatten}(X) = X.reshape(X.shape[0], -1)
    
    Args:
        X (np.ndarray): input array
        dL (np.ndarray, optional): backpropagated loss. Defaults to None.
        
    Returns:
        np.ndarray: flattening or backpropagated loss
    """
    if dL is not None:
        # compute backpropagated loss
        return dL.reshape(X.shape)
    else:
        # compute flattening
        return X.reshape(X.shape[0], -1)

    
def minibatch_gd_mnist_cnn(X, y, batch_size=32, epochs=10, lr=0.01):
    # CNN
    k_1 = np.random.randn(16, 1, 3, 3)
    k_2 = np.random.randn(32, 16, 3, 3)
    k_3 = np.random.randn(10, 32, 3, 3)
    
    W_1 = np.random.randn(4840, 10)

    for n in range(epochs):
        for i in trange(0, len(X), batch_size):
            x = X[i:i + batch_size]
            y_ = y[i:i + batch_size]
            
            conv1 = conv2d(x, k_1)
            relu1 = relu(conv1)
            conv2 = conv2d(relu1, k_2)
            relu2 = relu(conv2)
            conv3 = conv2d(relu2, k_3)
            relu3 = relu(conv3)
            
            # flatten
            x_fl = flatten(relu3)
            
            # dense
            x_mm = matmul(x_fl, W_1)
            
            # softmax
            y_hat = softmax(x_mm, dim=1)
            
            # loss
            loss = bce(y_hat, y_)
            
            # backpropagation
            dL = bce(y_hat, y_, dL=1)
            dL, dW_1 = matmul(x_fl, W_1, dL=dL)
            dL = flatten(relu3, dL)
            dL = relu(conv3, dL)
            dL, dK_3 = conv2d(relu2, k_3, dL=dL)
            dL = relu(conv2, dL)
            dL, dK_2 = conv2d(relu1, k_2, dL=dL)
            dL = relu(conv1, dL)
            dL, dK_1 = conv2d(x, k_1, dL=dL)
            
            # update
            k_1 -= lr * dK_1
            k_2 -= lr * dK_2
            k_3 -= lr * dK_3
            W_1 -= lr * dW_1
            
            print(f'Epoch {n + 1} | Loss: {loss:.4f}')
            
    def eval(x):
        conv1 = conv2d(x, k_1)
        relu1 = relu(conv1)
        conv2 = conv2d(relu1, k_2)
        relu2 = relu(conv2)
        conv3 = conv2d(relu2, k_3)
        relu3 = relu(conv3)
        
        # flatten
        x = relu3.reshape(relu3.shape[0], -1)
        
        # dense
        x = matmul(x, W_1)
        
        # softmax
        y_hat = softmax(x, dim=1)
        
        return y_hat
        
    return eval
    


if __name__ == '__main__':
    
    X = np.random.rand(1, 1, 28, 28)
    k = np.random.rand(1, 1, 3, 3)
    
    my_conv = conv2d(X, k)
    torch_conv = torch.nn.Conv2d(1, 1, 3, bias=False)
    torch_conv.weight.data = torch.tensor(k)
    torch_conv_out = torch_conv(torch.tensor(X))
    
    assert np.allclose(my_conv, torch_conv_out.detach().numpy())
    
    my_conv_dL = conv2d(X, k, dL=np.ones_like(my_conv))
    torch_conv_out.backward(torch.ones_like(torch_conv_out))
    
    assert np.allclose(my_conv_dL[1], torch_conv.weight.grad.detach().numpy())
    
    
    X, y, x_test, y_test = load_mnist()
    X = X.reshape(-1, 1, 28, 28).astype(np.float32)
    x_test = x_test.reshape(-1, 1, 28, 28).astype(np.float32)
    y = y.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    model = minibatch_gd_mnist_cnn(X, y, batch_size=32, epochs=10, lr=0.01)
    
    y_hat = model(x_test)
    
    print(f'Accuracy: {np.mean(np.argmax(y_hat, axis=1) == y_test)}')