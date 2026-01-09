import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings


warnings.filterwarnings('ignore')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOOKBACK = 7      
TARGET_STOCK = 6    
HIDDEN_SIZE = 64   
NUM_LAYERS = 1      
LEARNING_RATE = 0.005 
EPOCHS = 100        
BATCH_SIZE = 64     
N_ENSEMBLE = 50     # 集成模型数量
ALPHA = 0.05        # 95% 置信区间

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
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

def make_dataset(price_matrix, input_stocks, target_stock, lookback):
    n_steps = price_matrix.shape[1]
    X_list, y_list, idx_list = [], [], []
    for t in range(lookback, n_steps):
        feat = price_matrix[input_stocks, t - lookback:t].T 
        X_list.append(feat)
        y_list.append(price_matrix[target_stock, t])
        idx_list.append(t)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1), np.array(idx_list)

def train_ensemble_case(case_name, price_matrix, input_stocks):
    print(f"\n[{case_name}] 启动集成训练 (n={N_ENSEMBLE})...")
    

    X, y, idxs = make_dataset(price_matrix, input_stocks, TARGET_STOCK, LOOKBACK)
    
    # 划分训练/测试 (80/20)
    train_size = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:train_size], X[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:]
    test_idxs = idxs[train_size:]
    
    # 标准化 (X 和 y 分别标准化)
    N, T, F = X_train_raw.shape
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_s = scaler_x.fit_transform(X_train_raw.reshape(-1, F)).reshape(N, T, F)
    y_train_s = scaler_y.fit_transform(y_train_raw)
    
    # 测试集 Transform
    N_test, _, _ = X_test_raw.shape
    X_test_s = scaler_x.transform(X_test_raw.reshape(-1, F)).reshape(N_test, T, F)
    
    X_train_tensor = torch.from_numpy(X_train_s).to(DEVICE)
    y_train_tensor = torch.from_numpy(y_train_s).to(DEVICE)
    X_test_tensor = torch.from_numpy(X_test_s).to(DEVICE)

    # 集成循环 
    ensemble_preds = []      # 存储每个模型对测试集的预测
    oob_abs_residuals = []   # 存储所有 OOB 样本的绝对残差
    
    n_train_samples = X_train_s.shape[0]
    indices = np.arange(n_train_samples)
    
    for i in range(N_ENSEMBLE):
        # Bootstrap 采样
        boot_idx = np.random.choice(indices, size=n_train_samples, replace=True)
        # OOB 索引
        oob_mask = np.ones(n_train_samples, dtype=bool)
        oob_mask[boot_idx] = False
        oob_idx = indices[oob_mask]
        
        X_boot = X_train_tensor[boot_idx]
        y_boot = y_train_tensor[boot_idx]
        
        train_ds = TensorDataset(X_boot, y_boot)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        model = StockLSTM(input_size=F, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(EPOCHS):
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            # 预测 OOB (用于不确定性)
            if len(oob_idx) > 0:
                X_oob = X_train_tensor[oob_idx]
                y_oob_true = y_train_raw[oob_idx]
                
                pred_oob_s = model(X_oob).cpu().numpy()
                pred_oob = scaler_y.inverse_transform(pred_oob_s)
                
                residuals = np.abs(pred_oob - y_oob_true)
                oob_abs_residuals.extend(residuals.flatten())
            
            # 预测测试集
            pred_test_s = model(X_test_tensor).cpu().numpy()
            pred_test = scaler_y.inverse_transform(pred_test_s)
            ensemble_preds.append(pred_test)
            
        print(f"  Model {i+1}/{N_ENSEMBLE} done.", end="\r")

    print(f"\n  Training finished.")
    
    ensemble_preds = np.array(ensemble_preds) 
    final_pred = np.mean(ensemble_preds, axis=0)
    pred_std = np.std(ensemble_preds, axis=0)
    
    if len(oob_abs_residuals) > 0:
        q_conformal = np.quantile(oob_abs_residuals, 1.0 - ALPHA)
    else:
        q_conformal = 1.96 * np.mean(pred_std)
        
    ci_lower = final_pred - q_conformal
    ci_upper = final_pred + q_conformal
    
    mse = mean_squared_error(y_test_raw, final_pred)
    mae = mean_absolute_error(y_test_raw, final_pred)
    r2 = r2_score(y_test_raw, final_pred)
    
    std_mean = np.mean(pred_std)  
    std_max = np.max(pred_std)   
    
    print(f"  R2: {r2:.4f} | MSE: {mse:.4f} | Conf_q: {q_conformal:.4f} | StdMean: {std_mean:.4f} | StdMax: {std_max:.4f}")
    
    return {
        "case": case_name,
        "test_idx": test_idxs,
        "y_true": y_test_raw,
        "y_pred": final_pred,
        "pred_std": pred_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "q_conformal": q_conformal,
        "std_mean": std_mean,
        "std_max": std_max,  
        "metrics": (mse, mae, r2)
    }


if __name__ == "__main__":
    try:
        prices = np.loadtxt("prices.csv", delimiter=",")
    except Exception as e:
        print(f"Error: {e}")
        exit()

    results = []

    # Case 1
    results.append(train_ensemble_case("Stocks 1-6 -> Stock 7", prices, list(range(6))))
    
    # Case 2
    results.append(train_ensemble_case("Stock 7 -> Stock 7", prices, [6]))
    
    # Case 3
    results.append(train_ensemble_case("All 7 -> Stock 7", prices, list(range(7))))

    # 绘图 1

    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), dpi=120) 
    

    COLOR_TRUE = '#2C3E50'      
    COLOR_PRED = '#C0392B'    
    COLOR_BAND = '#2980B9'      
    
    for idx, res in enumerate(results):
        ax = axes[idx]
        t = res['test_idx']
        y_true = res['y_true']
        y_pred = res['y_pred']
        lower = res['ci_lower'].flatten()
        upper = res['ci_upper'].flatten()
        mse, mae, r2 = res['metrics']
        
        # 计算平均区间宽度 (Mean Prediction Interval Width)
        mpiw = np.mean(upper - lower)
        
        # 绘制置信区间 (最底层 zorder=1)

        ax.fill_between(t, lower, upper, color=COLOR_BAND, alpha=0.15, 
                        label=f'95% Conformal CI (Width: {mpiw:.2f})', zorder=1)
        
        # 绘制区间边界线 
        ax.plot(t, lower, color=COLOR_BAND, alpha=0.3, linewidth=0.5, zorder=1)
        ax.plot(t, upper, color=COLOR_BAND, alpha=0.3, linewidth=0.5, zorder=1)

        # 绘制预测均值 
        ax.plot(t, y_pred, label='Ensemble Prediction', color=COLOR_PRED, 
                linewidth=1.8, linestyle='-', zorder=2)

        # 绘制真实价格
        ax.plot(t, y_true, label='True Price', color=COLOR_TRUE, 
                linewidth=1.2, alpha=0.9, zorder=3)

        ax.set_title(f"Case {idx+1}: {res['case']} | R²={r2:.3f} | MAE={mae:.3f} | Mean Interval Width={mpiw:.2f}", 
                     fontsize=11, fontweight='bold', pad=10)
        
        ax.set_ylabel("Normalized Price / Price")
        ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9, fontsize=9)
        ax.grid(True)
        

        ax.margins(x=0.01)

    axes[-1].set_xlabel("Time Step (Days)", fontsize=11)

    fig.suptitle('LSTM Ensemble Strategy Comparison: Accuracy vs Uncertainty', fontsize=14, y=0.99)
    
    plt.tight_layout()
    plt.savefig('lstm_ensemble_overview_refined.png')
    plt.show()
    print("概览图绘图完成，已保存为 lstm_ensemble_overview_refined.png")


    import scipy.stats as stats
    
    res_demo = results[1]  # 使用 Case 2
    
    # 添加 .flatten() 确保变为一维数组 
    t = res_demo['test_idx'].flatten()
    y_true = res_demo['y_true'].flatten()
    y_pred = res_demo['y_pred'].flatten()
    std = res_demo['pred_std'].flatten()
    
    # 这里的 q_conformal 是标量，不需要 flatten
    q_adj = res_demo['q_conformal'] 
    

    fig = plt.figure(figsize=(14, 10), dpi=120)
    gs = fig.add_gridspec(3, 2) 
    
    # 金融扇形图 (Fan Chart) 
    ax_main = fig.add_subplot(gs[0:2, :]) 
    
    # 1. 绘制最外层：基于共形预测的 95% 区间 
    ax_main.fill_between(t, 
                         y_pred - q_adj, 
                         y_pred + q_adj, 
                         color='gray', alpha=0.15, 
                         label=f'Conformal 95% Interval (Robust to Fat-tails)')
    
    # 2. 绘制内层扇形：基于模型标准差的渐进概率 (假设局部高斯分布)
    # 50% 置信区间 (±0.67 std)
    ax_main.fill_between(t, y_pred - 0.67*std, y_pred + 0.67*std, color='#1f77b4', alpha=0.5, label='50% CI (Model Uncertainty)')
    # 80% 置信区间 (±1.28 std)
    ax_main.fill_between(t, y_pred - 1.28*std, y_pred + 1.28*std, color='#1f77b4', alpha=0.3, label='80% CI')
    # 95% 置信区间 (±1.96 std)
    ax_main.fill_between(t, y_pred - 1.96*std, y_pred + 1.96*std, color='#1f77b4', alpha=0.15, label='95% CI (Normal Assumption)')

    # 3. 绘制线条
    ax_main.plot(t, y_true, color='black', lw=1.5, label='True Price')
    ax_main.plot(t, y_pred, color='#004c8c', lw=2, linestyle='--', label='Ensemble Mean')
    
    ax_main.set_title(f"Financial Fan Chart: Prediction with Uncertainty Layers (Case: {res_demo['case']})", fontsize=14, fontweight='bold')
    ax_main.set_ylabel("Stock Price")
    ax_main.legend(loc='upper left', frameon=True, shadow=True)
    ax_main.grid(True, alpha=0.3)
    
    # 局部缩放 (Zoom In)
    ax_zoom = fig.add_subplot(gs[2, 0])
    zoom_slice = slice(-30, None)
    
    ax_zoom.plot(t[zoom_slice], y_true[zoom_slice], 'k.-', label='True')
    ax_zoom.plot(t[zoom_slice], y_pred[zoom_slice], 'b--', label='Pred')
    
    ax_zoom.fill_between(t[zoom_slice], 
                         (y_pred - q_adj)[zoom_slice], 
                         (y_pred + q_adj)[zoom_slice], 
                         color='gray', alpha=0.2)
    ax_zoom.set_title("Zoom-in (Last 30 Steps)", fontsize=10)
    ax_zoom.grid(True, alpha=0.3)
    
    # 肥尾检验 (Fat-tail Check)
    ax_hist = fig.add_subplot(gs[2, 1])
    
    # 计算测试集残差
    residuals = y_true - y_pred
    
    # 绘制残差直方图
    ax_hist.hist(residuals, bins=30, density=True, alpha=0.6, color='orange', label='Actual Residuals')
    
    # 拟合正态分布曲线进行对比
    mu, sigma = stats.norm.fit(residuals)
    xmin, xmax = ax_hist.get_xlim()
    x_grid = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x_grid, mu, sigma)
    ax_hist.plot(x_grid, p, 'k', linewidth=2, label='Normal Distribution Fit')
    
    ax_hist.set_title("Fat-tail Analysis: Residuals vs Normal", fontsize=10)
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_financial_plot.png')
    plt.show()
    print("高级绘图完成，已保存为 advanced_financial_plot.png")