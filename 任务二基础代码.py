import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOOKBACK = 7        # 过去7天
TARGET_STOCK = 6    # 预测第7支股票 (索引为6)
HIDDEN_SIZE = 32    # LSTM 隐藏层大小
NUM_LAYERS = 1      # LSTM 层数
LEARNING_RATE = 0.001
EPOCHS = 200
BATCH_SIZE = 32


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播 LSTM
        # out 的形状: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出用于预测
        out = out[:, -1, :] 
        
        # 通过全连接层
        out = self.fc(out)
        return out

def make_dataset(price_matrix, input_stocks, target_stock, lookback):
    """
    创建适合 RNN/LSTM 的数据集。
    输入 X 形状: (样本数, lookback, 特征数)
    输出 y 形状: (样本数, 1)
    """
    n_steps = price_matrix.shape[1]
    X_list, y_list, idx_list = [], [], []
    
    for t in range(lookback, n_steps):
        # 提取输入特征：形状为 (特征数, lookback) -> 转置为 (lookback, 特征数)
        feat = price_matrix[input_stocks, t - lookback:t].T 
        X_list.append(feat)
        
        # 提取目标值
        y_list.append(price_matrix[target_stock, t])
        idx_list.append(t)
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1), np.array(idx_list)

def train_eval_case_pytorch(case_name, price_matrix, input_stocks):
    print(f"\n启动任务: {case_name}")
    
    # 数据准备 
    X, y, idxs = make_dataset(price_matrix, input_stocks, TARGET_STOCK, LOOKBACK)
    
    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:train_size], X[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:] # y 也保留一份 raw 用于验证
    test_idxs = idxs[train_size:]
    
    # 对 X 和 y 分别进行标准化 
    N, T, F = X_train_raw.shape
    
    # Scaler
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit 并在训练集上 Transform
    # X 需要 reshape 才能 fit
    X_train_s = scaler_x.fit_transform(X_train_raw.reshape(-1, F)).reshape(N, T, F)
    y_train_s = scaler_y.fit_transform(y_train_raw) 
    
    # 在测试集上 Transform (只用 transform，不要 fit)
    N_test, _, _ = X_test_raw.shape
    X_test_s = scaler_x.transform(X_test_raw.reshape(-1, F)).reshape(N_test, T, F)
    # y_test 不需要 transform，因为我们要用逆变换后的预测值去和真实的 y_test_raw 比较
    
    X_train_tensor = torch.from_numpy(X_train_s).to(DEVICE)
    y_train_tensor = torch.from_numpy(y_train_s).to(DEVICE)
    X_test_tensor = torch.from_numpy(X_test_s).to(DEVICE)
    

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    

    model = StockLSTM(input_size=F, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    for epoch in range(EPOCHS):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) # 这里的 loss 是在标准化尺度下计算的
            loss.backward()
            optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(X_test_tensor).cpu().numpy()
    

    test_preds = scaler_y.inverse_transform(test_preds_scaled)
    
    mse = mean_squared_error(y_test_raw, test_preds)
    mae = mean_absolute_error(y_test_raw, test_preds)
    r2 = r2_score(y_test_raw, test_preds)
    
    print(f"[{case_name}] 结果: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
    return {
        "case": case_name,
        "y_test": y_test_raw, 
        "pred": test_preds, 
        "test_idx": test_idxs,
        "metrics": (mse, mae, r2)
    }

if __name__ == "__main__":
    try:
        prices = np.loadtxt("prices.csv", delimiter=",")
        print(f"成功读取数据，形状: {prices.shape}")
    except Exception as e:
        print(f"读取数据失败: {e}")
        print("请确保 prices.csv 在当前目录下")
        exit()

    results = []

    # 任务 1: 输入前 6 支 -> 预测第 7 支
    results.append(train_eval_case_pytorch(
        "Stocks 1-6 -> Stock 7", 
        prices, 
        input_stocks=[0, 1, 2, 3, 4, 5]
    ))

    # 任务 2: 输入第 7 支 -> 预测第 7 支
    results.append(train_eval_case_pytorch(
        "Stock 7 -> Stock 7", 
        prices, 
        input_stocks=[6]
    ))

    # 任务 3: 输入全部 7 支 -> 预测第 7 支
    results.append(train_eval_case_pytorch(
        "All 7 Stocks -> Stock 7", 
        prices, 
        input_stocks=[0, 1, 2, 3, 4, 5, 6]
    ))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=100)
    
    for idx, res in enumerate(results):
        ax = axes[idx]
        t = res['test_idx']
        y_true = res['y_test']
        y_pred = res['pred']
        mse, mae, r2 = res['metrics']
        
        ax.plot(t, y_true, label='True Price', color='black', alpha=0.7)
        ax.plot(t, y_pred, label='LSTM Prediction', color='blue', linestyle='--')
        
        ax.set_title(f"{res['case']} (R2={r2:.3f}, MAE={mae:.3f})")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.savefig('lstm_prediction_results.png') 
    plt.show()