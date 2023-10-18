import numpy as np
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class DataLoader(object):
    def __init__(self, file, horizon, window, missing_value=-9999):

        self.window = window
        self.horizon = horizon
        
        self.data = np.load(f'data/2018/{file}.npy', allow_pickle=True)
        self.missing_value = missing_value
        self.sample_num, self.feature_num = self.data.shape
        self.maxFeatureValue = np.max(self.data, axis=0) + 1
        # 保存self.maxFeatureValue到文件
        #np.save('max_feature_value.npy', self.maxFeatureValue)
        X, Y, C = self._getXY()
        self._split(X, Y, C)
        self.y_scale = self.maxFeatureValue[0]

    def _split(self, X, Y, C):
        train_indices, val_indices, test_indices = split_label(C) 

        self.X_train, self.y_train = X[train_indices,:,:], Y[train_indices,:]
        self.X_valid, self.y_valid = X[val_indices,:,:], Y[val_indices,:]
        self.X_test, self.y_test = X[test_indices,:,:], Y[test_indices,:]
        
        print(self.X_train.shape, self.y_train.shape)
        print(self.X_valid.shape, self.y_valid.shape)
        print(self.X_test.shape, self.y_test.shape)

    def _getXY(self):
        X_Y_pair_num = self.sample_num - self.window - self.horizon + 1
        X = np.zeros((X_Y_pair_num, self.window, self.feature_num))
        Y = np.zeros((X_Y_pair_num, self.horizon))
        C = np.zeros((X_Y_pair_num))
        print(X.shape, Y.shape)
        
        indices = []
        X_start = 0
        Y_start = self.window
        for i in range(0, X_Y_pair_num):
            if np.any(self.data[Y_start:Y_start+self.horizon, 0] == self.missing_value):
                indices.append(i) # 记录位置
            else:
                # 标准化 分别除以每一列的最大值
                x = self.data[X_start:X_start+self.window, :] / self.maxFeatureValue[:]
                y = self.data[Y_start:Y_start+self.horizon, 0]
                c = to_class(y)
                y = y / self.maxFeatureValue[0]

                X[i, :, :] = x
                Y[i, :] = y
                C[i] = max(c)

            X_start += 1
            Y_start += 1
        
        # 删除位置
        X = np.delete(X, indices, axis=0)
        Y = np.delete(Y, indices, axis=0)
        C = np.delete(C, indices, axis=0)
        print(X.shape, Y.shape)
        return X, Y, C    


class DataLoader2(object):
    def __init__(self, file, horizon, window, missing_value=-9999):

        self.window = window
        self.horizon = horizon
        
        self.data = np.load(f'data/2019/{file}.npy', allow_pickle=True)
        self.missing_value = missing_value
        self.sample_num, self.feature_num = self.data.shape
        #self.maxFeatureValue = np.max(self.data, axis=0) + 1
        # 在需要时加载self.maxFeatureValue
        self.maxFeatureValue = np.load('max_feature_value.npy')
        self.X, self.Y, _ = self._getXY()
        self.y_scale = self.maxFeatureValue[0]

    def _getXY(self):
        X_Y_pair_num = self.sample_num - self.window - self.horizon + 1
        X = np.zeros((X_Y_pair_num, self.window, self.feature_num))
        Y = np.zeros((X_Y_pair_num, self.horizon))
        C = np.zeros((X_Y_pair_num))
        print(X.shape, Y.shape)
        
        indices = []
        X_start = 0
        Y_start = self.window
        for i in range(0, X_Y_pair_num):
            if np.any(self.data[Y_start:Y_start+self.horizon, 0] == self.missing_value):
                indices.append(i) # 记录位置
            else:
                # 标准化 分别除以每一列的最大值
                x = self.data[X_start:X_start+self.window, :] / self.maxFeatureValue[:]
                y = self.data[Y_start:Y_start+self.horizon, 0]
                c = to_class(y)
                y = y / self.maxFeatureValue[0]

                X[i, :, :] = x
                Y[i, :] = y
                C[i] = max(c)

            X_start += 1
            Y_start += 1
        
        # 删除位置
        X = np.delete(X, indices, axis=0)
        Y = np.delete(Y, indices, axis=0)
        C = np.delete(C, indices, axis=0)
        print(X.shape, Y.shape)
        return X, Y, C


def get_batches(X, Y, batch_size, shuffle=True):
    length = len(X)
    if shuffle:
        index = torch.randperm(length)
    else:
        index = torch.LongTensor(range(length))
    start_idx = 0
    while (start_idx < length):
        end_idx = min(length, start_idx + batch_size)
        excerpt = index[start_idx:end_idx]
        tx = X[excerpt]
        ty = Y[excerpt]
        
        yield Variable(tx), Variable(ty)
        start_idx += batch_size

def to_class(x):
    conditions = [x > 10000,
                  x > 1000,
                  x > 500,
                  x > 200,
                  x > 50]
    choices = [0, 1, 2, 3, 4]
    result = np.select(conditions, choices, 5)

    return result

def test1():
    x = np.array([200, 1000, 5000])

    result = to_class(x)

    print(result)

def split_label(labels: np.array, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    # Ensure the sum of ratios equals 1
    assert train_ratio + val_ratio + test_ratio == 1

    # Set the random seed
    np.random.seed(random_seed)

    def split_indices(indices, train_size, val_size):
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        return train_indices, val_indices, test_indices

    # Get the indices for each class
    class_indices = []
    for class_label in np.unique(labels):
        indices = np.where(labels == class_label)[0]
        np.random.shuffle(indices)
        class_indices.append(indices)

    # Calculate the number of samples for each class
    class_sample_nums = [len(indices) for indices in class_indices]

    # Calculate the number of samples for each class in train and val sets
    train_sizes = [int(train_ratio * num) for num in class_sample_nums]
    val_sizes = [int(val_ratio * num) for num in class_sample_nums]

    # Split the indices for each class
    train_indices = []
    val_indices = []
    test_indices = []
    for indices, train_size, val_size in zip(class_indices, train_sizes, val_sizes):
        class_train_indices, class_val_indices, class_test_indices = split_indices(indices, train_size, val_size)
        train_indices.append(class_train_indices)
        val_indices.append(class_val_indices)
        test_indices.append(class_test_indices)

    # Concatenate the indices for each set
    train_indices = np.concatenate(train_indices)
    val_indices = np.concatenate(val_indices)
    test_indices = np.concatenate(test_indices)

    # Shuffle the indices for each set
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    class_labels = np.unique(labels)
    set_name = ['Train', 'Valid', 'Test']
    for i, indices in enumerate([train_indices, val_indices, test_indices]):
        print(set_name[i], "Set")
        for class_label in class_labels:
            class_indices = indices[labels[indices] == class_label]
            class_sample_count = len(class_indices)
            print("Class", class_label)
            print("Sample Count:", class_sample_count)
            # print("Indices:", class_indices)
            print()
        print("==========")

    return train_indices, val_indices, test_indices

def test_4_split_label():
    a = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

    split_label(a)

def acc(y_pred, y_true):
    assert len(y_true) == len(y_pred), 'length not match'

    length = len(y_true)

    y_true_level = to_class(y_true)
    y_pred_level = to_class(y_pred)

    probability = np.sum(y_true_level == y_pred_level) / length

    print(f'样本总数{length}, 所有等级相同的概率{probability:.2%}')
        
    # 计算不同等级相同的概率
    for level in range(0, 6):
        same_count = np.sum((y_true_level == level) & (y_pred_level == level))
        total_count = np.sum(y_true_level == level)
        probability = same_count / total_count if total_count > 0 else -1
        print(f'等级{level}, 样本数{total_count}, 预测对的样本数{same_count}, 相同的概率{probability:.2%}')


def print_metrics(y_pred, y_true):
    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)
    dic = {}
    
    # Mean Absolute Percentage Error
    dic['MAPE'] = np.mean(np.abs((y_pred-y_true) / y_true))

    # R-Square
    dic['R2'] = r2_score(y_true, y_pred)

    # Root Mean Squared Error
    dic['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    dic['Corr'] = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Mean Absolute Error
    dic['MAE'] = mean_absolute_error(y_true, y_pred)

    # Sum of Squared Errors
    dic['SSE'] = np.sum(np.square(y_true-y_pred))      

    for key, value in dic.items():
        print("{}: {:.4f}".format(key, value))
    
    acc(y_pred, y_true)


def draw_loss(loss_values, epoch_num):
    plt.plot(range(epoch_num), loss_values, marker='o', linestyle='-', color='b')

    # 设置标题和坐标轴标签
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 显示网格线
    plt.grid(True)

    # 显示图形
    plt.show()


def draw(y_true, y_pred):
    # 绘制折线图
    x = np.arange(y_true.shape[0])  # 假设x轴是索引
    plt.plot(x, y_true, label='Actual')
    plt.plot(x, y_pred, label='Predict')

    # 添加图例和标签
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')

    # 显示图像
    plt.show()