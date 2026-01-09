import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子保证结果可复现
np.random.seed(42)
torch.manual_seed(42)


def lorenz(t, xyz):
    x, y, z = xyz
    s, r, b = 10, 28, 8 / 3.0
    return [s * (y - x), x * (r - z) - y, x * y - b * z]


a, b = 0, 40
t_total = np.linspace(a, b, 4000)
sol = solve_ivp(lorenz, [a, b], [1, 1, 1], t_eval=t_total)
data = sol.y.T  # 转置为 (4000, 3)，每行是 (x, y, z)

print(f"Original Data Shape: {data.shape}")


train_size = 3000
test_size = 1000

# 原始数据切分
train_raw = data[:train_size]      # (3000, 3)
test_raw = data[train_size:]       # (1000, 3)

# 提取特征
scaler = StandardScaler()
train_xy_scaled = scaler.fit_transform(train_raw[:, :2])
test_xy_scaled = scaler.transform(test_raw[:, :2])

train_z = train_raw[:, 2]
test_z = test_raw[:, 2]


lookback = 7

# 拼接 Train 末尾 7 个点 + Test 所有点
test_xy_extended = np.vstack([train_xy_scaled[-lookback:], test_xy_scaled])

# 拼接 Train 末尾 7 个点 + Test 所有点
# 保持和 X 长度一致 (1007)
test_z_extended = np.concatenate([train_z[-lookback:], test_z])


def create_dataset(xy_data, z_data, lookback):
    """
    构造 LSTM 需要的 3D 输入格式: (Sample, TimeStep, Feature)
    """
    X, y = [], []
    for i in range(len(xy_data) - lookback):
        # i=0 时: 取 xy_data[0:7] (即历史7个点), 目标是 z_data[7] (即 test_z 的第0个点)
        X.append(xy_data[i : i + lookback])
        y.append(z_data[i + lookback]) 
        
    return np.array(X), np.array(y)

# 构造数据
X_train, y_train = create_dataset(train_xy_scaled, train_z, lookback)

X_test, y_test = create_dataset(test_xy_extended, test_z_extended, lookback)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

print(f"Train samples: {X_train_t.shape}, Test samples: {X_test_t.shape}")
# 输出应为: Test samples: torch.Size([1000, 7, 2])
# 预期 Train: (2993, 7, 2), Test: (1000, 7, 2)

class LorenzLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=1, num_layers=1):
        super(LorenzLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True 使得输入维度为 (batch, seq_len, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层将 LSTM 输出映射到 z 值
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM 输出: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x)
        
        # 只关心序列最后一个时间步的输出用于预测
        last_time_step = out[:, -1, :] 
        
        prediction = self.fc(last_time_step)
        return prediction

model = LorenzLSTM(input_size=2, hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

epochs = 200
loss_history = []

model.train()
print("Start Training...")
for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    loss_history.append(epoch_loss / len(train_loader))
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_history[-1]:.6f}")


model.eval()
with torch.no_grad():
    test_pred_t = model(X_test_t)
    train_pred_t = model(X_train_t)

test_pred = test_pred_t.numpy().flatten()
y_test_np = y_test_t.numpy().flatten()

mse = mean_squared_error(y_test_np, test_pred)
mae = mean_absolute_error(y_test_np, test_pred)
r2 = r2_score(y_test_np, test_pred)

print("-" * 30)
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R^2: {r2:.4f}")


fig = plt.figure(figsize=(14, 10), dpi=150)
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# 图 1: 全局时序对比
ax1 = plt.subplot(2, 2, 1)
time_steps = np.arange(len(y_test_np))
ax1.plot(time_steps, y_test_np, label='True z', color='black', alpha=0.7, lw=1)
ax1.plot(time_steps, test_pred, label='LSTM Prediction', color='#E63946', alpha=0.8, lw=1)
ax1.set_title('Global Forecast (Test Set)', fontweight='bold')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('z Value')
ax1.legend(loc='upper right', frameon=False)
ax1.grid(True, alpha=0.2)
ax1.text(0.02, 0.95, f"R²={r2:.4f}", transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

# 图 2: 局部放大 (Zoom-in)
zoom_slice = slice(100, 200)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(time_steps[zoom_slice], y_test_np[zoom_slice], '.-', label='True z', color='black', lw=1.5)
ax2.plot(time_steps[zoom_slice], test_pred[zoom_slice], '.-', label='Prediction', color='#E63946', lw=1.5)
ax2.set_title(f'Zoom-in View (Steps 100-200)', fontweight='bold')
ax2.set_xlabel('Time Step')
ax2.grid(True, alpha=0.2)
ax2.legend(loc='upper right')

# 图 3: 回归散点图
ax3 = plt.subplot(2, 2, 3)
ax3.scatter(y_test_np, test_pred, alpha=0.5, s=10, color='#457B9D')
# 45度参考线
min_val = min(y_test_np.min(), test_pred.min())
max_val = max(y_test_np.max(), test_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
ax3.set_title('Prediction vs Ground Truth', fontweight='bold')
ax3.set_xlabel('True z')
ax3.set_ylabel('Predicted z')
ax3.grid(True, alpha=0.2)

# 图 4: 误差分布直方图
residuals = test_pred - y_test_np
ax4 = plt.subplot(2, 2, 4)
ax4.hist(residuals, bins=50, color='#1D3557', alpha=0.7, density=True)
ax4.axvline(0, color='r', linestyle='--', lw=1)
ax4.set_title('Error Distribution (Residuals)', fontweight='bold')
ax4.set_xlabel('Error (Pred - True)')
ax4.set_ylabel('Density')
ax4.grid(True, alpha=0.2)

plt.suptitle(f"Lorenz System Prediction using PyTorch LSTM\n(Lookback={lookback}, Inputs: x,y -> Output: z)", fontsize=14)
plt.savefig('lorenz_lstm_analysis.png', bbox_inches='tight')
plt.show()