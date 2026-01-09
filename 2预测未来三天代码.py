import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
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
N_ENSEMBLE = 50    
ALPHA = 0.05        

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # output_size 决定了是预测1天还是3天
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

def make_dataset(price_matrix, input_stocks, target_stock, lookback, horizon=1):
    n_steps = price_matrix.shape[1]
    X_list, y_list, idx_list = [], [], []
    

    for t in range(lookback, n_steps - horizon + 1):
        # 输入：t-lookback 到 t (不包含 t)
        feat = price_matrix[input_stocks, t - lookback : t].T 
        # 输出：t 到 t+horizon (包含 t, t+1... t+horizon-1)
        target = price_matrix[target_stock, t : t + horizon]
        
        X_list.append(feat)
        y_list.append(target)
        # 记录目标时间点的索引（我们记录预测序列的最后一天作为时间戳）
        idx_list.append(t + horizon - 1)
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), np.array(idx_list)


def train_ensemble_case(case_name, price_matrix, input_stocks, horizon=1):
    print(f"\n[{case_name}] 启动训练 (Horizon={horizon} days, n={N_ENSEMBLE})...")
    
    # y 的形状现在是 (N, horizon)
    X, y, idxs = make_dataset(price_matrix, input_stocks, TARGET_STOCK, LOOKBACK, horizon)
    
    # 划分训练/测试 (80/20)
    train_size = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:train_size], X[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:]
    test_idxs = idxs[train_size:]
    
    # 标准化 
    N, T, F = X_train_raw.shape
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_s = scaler_x.fit_transform(X_train_raw.reshape(-1, F)).reshape(N, T, F)
    y_train_s = scaler_y.fit_transform(y_train_raw) # (N, horizon)
    
    N_test, _, _ = X_test_raw.shape
    X_test_s = scaler_x.transform(X_test_raw.reshape(-1, F)).reshape(N_test, T, F)
    
    X_train_tensor = torch.from_numpy(X_train_s).to(DEVICE)
    y_train_tensor = torch.from_numpy(y_train_s).to(DEVICE)
    X_test_tensor = torch.from_numpy(X_test_s).to(DEVICE)

    ensemble_preds = []      
    oob_abs_residuals = []   
    
    n_train_samples = X_train_s.shape[0]
    indices = np.arange(n_train_samples)
    
    for i in range(N_ENSEMBLE):
        boot_idx = np.random.choice(indices, size=n_train_samples, replace=True)
        oob_mask = np.ones(n_train_samples, dtype=bool)
        oob_mask[boot_idx] = False
        oob_idx = indices[oob_mask]
        
        X_boot = X_train_tensor[boot_idx]
        y_boot = y_train_tensor[boot_idx]
        
        train_ds = TensorDataset(X_boot, y_boot)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        model = StockLSTM(input_size=F, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=horizon).to(DEVICE)
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
            # OOB 预测 (用于计算共形区间)
            if len(oob_idx) > 0:
                X_oob = X_train_tensor[oob_idx]
                y_oob_true = y_train_raw[oob_idx] 
                
                pred_oob_s = model(X_oob).cpu().numpy()
                pred_oob = scaler_y.inverse_transform(pred_oob_s)
                

                # 取 horizon 的最后一天来计算残差，代表"预测第N天的风险"
                residuals = np.abs(pred_oob[:, -1] - y_oob_true[:, -1])
                oob_abs_residuals.extend(residuals.flatten())
            
            # 测试集预测
            pred_test_s = model(X_test_tensor).cpu().numpy()
            pred_test = scaler_y.inverse_transform(pred_test_s)
            ensemble_preds.append(pred_test)
            
        print(f"  Model {i+1}/{N_ENSEMBLE} done.", end="\r")

    print(f"\n  Training finished.")
    

    ensemble_preds = np.array(ensemble_preds) # Shape: (n_models, N_test, horizon)
    
    # 只取出 Horizon 的最后一天进行可视化和评估
    final_pred_all = np.mean(ensemble_preds, axis=0) # (N_test, horizon)
    pred_std_all = np.std(ensemble_preds, axis=0)    # (N_test, horizon)
    
    # 提取最后一天的数据用于绘图
    final_pred_plot = final_pred_all[:, -1]
    pred_std_plot = pred_std_all[:, -1]
    y_test_plot = y_test_raw[:, -1]
    
    # 计算分位数 (基于 OOB 残差)
    if len(oob_abs_residuals) > 0:
        q_conformal = np.quantile(oob_abs_residuals, 1.0 - ALPHA)
    else:
        q_conformal = 1.96 * np.mean(pred_std_plot)
        
    ci_lower = final_pred_plot - q_conformal
    ci_upper = final_pred_plot + q_conformal
    
    mse = mean_squared_error(y_test_plot, final_pred_plot)
    mae = mean_absolute_error(y_test_plot, final_pred_plot)
    r2 = r2_score(y_test_plot, final_pred_plot)
    std_mean = np.mean(pred_std_plot)  
    std_max = np.max(pred_std_plot) 
    print(f"  R2: {r2:.4f} | MSE: {mse:.4f} | Conf_q: {q_conformal:.4f} | StdMean: {std_mean:.4f} | StdMax: {std_max:.4f}")
    return {
        "case": case_name,
        "horizon": horizon,
        "test_idx": test_idxs,
        "y_true": y_test_plot,
        "y_pred": final_pred_plot,
        "pred_std": pred_std_plot,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "q_conformal": q_conformal,
        "metrics": (mse, mae, r2)
    }


def plot_overview(results_list, horizon):
    """绘制三案例概览图"""
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), dpi=100)
    
    COLOR_TRUE = '#2C3E50'
    COLOR_PRED = '#C0392B'
    COLOR_BAND = '#2980B9'
    
    for idx, res in enumerate(results_list):
        ax = axes[idx]
        t = res['test_idx'].flatten()
        y_true = res['y_true'].flatten()
        y_pred = res['y_pred'].flatten()
        lower = res['ci_lower'].flatten()
        upper = res['ci_upper'].flatten()
        mse, mae, r2 = res['metrics']
        mpiw = np.mean(upper - lower)
        
        ax.fill_between(t, lower, upper, color=COLOR_BAND, alpha=0.15, 
                        label=f'95% Conformal CI (Width: {mpiw:.2f})', zorder=1)
        ax.plot(t, lower, color=COLOR_BAND, alpha=0.3, linewidth=0.5, zorder=1)
        ax.plot(t, upper, color=COLOR_BAND, alpha=0.3, linewidth=0.5, zorder=1)
        ax.plot(t, y_pred, label=f'T+{horizon} Prediction', color=COLOR_PRED, 
                linewidth=1.8, linestyle='-', zorder=2)
        ax.plot(t, y_true, label='True Price', color=COLOR_TRUE, 
                linewidth=1.2, alpha=0.9, zorder=3)
        
        ax.set_title(f"Case {idx+1}: {res['case']} | Horizon: +{horizon} Day(s) | R²={r2:.3f} | Width={mpiw:.2f}", 
                     fontsize=11, fontweight='bold', pad=10)
        
        ax.set_ylabel("Price")
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True)
        ax.margins(x=0.01)

    axes[-1].set_xlabel("Time Step (Days)", fontsize=11)
    fig.suptitle(f'LSTM Ensemble: {horizon}-Day Ahead Forecast Comparison', fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig(f'overview_horizon_{horizon}.png')
    plt.show()

def plot_fan_chart(res, horizon):
    """绘制详细扇形图"""
    t = res['test_idx'].flatten()
    y_true = res['y_true'].flatten()
    y_pred = res['y_pred'].flatten()
    std = res['pred_std'].flatten()
    q_adj = res['q_conformal'] 
    
    fig = plt.figure(figsize=(14, 10), dpi=100)
    gs = fig.add_gridspec(3, 2)
    
    # 主图
    ax_main = fig.add_subplot(gs[0:2, :])
    ax_main.fill_between(t, y_pred - q_adj, y_pred + q_adj, color='gray', alpha=0.15, 
                         label=f'Conformal 95% (Robust)')
    ax_main.fill_between(t, y_pred - 0.67*std, y_pred + 0.67*std, color='#1f77b4', alpha=0.5, label='50% CI')
    ax_main.fill_between(t, y_pred - 1.28*std, y_pred + 1.28*std, color='#1f77b4', alpha=0.3, label='80% CI')
    ax_main.fill_between(t, y_pred - 1.96*std, y_pred + 1.96*std, color='#1f77b4', alpha=0.15, label='95% CI (Normal)')

    ax_main.plot(t, y_true, color='black', lw=1.5, label='True Price')
    ax_main.plot(t, y_pred, color='#004c8c', lw=2, linestyle='--', label=f'T+{horizon} Mean Pred')
    
    ax_main.set_title(f"Fan Chart: {res['case']} (Horizon: +{horizon} Days)", fontsize=14, fontweight='bold')
    ax_main.set_ylabel("Price")
    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)
    
    # 局部缩放
    ax_zoom = fig.add_subplot(gs[2, 0])
    zoom_slice = slice(-40, None)
    ax_zoom.plot(t[zoom_slice], y_true[zoom_slice], 'k.-', label='True')
    ax_zoom.plot(t[zoom_slice], y_pred[zoom_slice], 'b--', label='Pred')
    ax_zoom.fill_between(t[zoom_slice], (y_pred - q_adj)[zoom_slice], (y_pred + q_adj)[zoom_slice], color='gray', alpha=0.2)
    ax_zoom.set_title(f"Zoom-in (Last 40 Days) - Lag Check", fontsize=10)
    ax_zoom.grid(True, alpha=0.3)
    
    # 残差分析
    ax_hist = fig.add_subplot(gs[2, 1])
    residuals = y_true - y_pred
    ax_hist.hist(residuals, bins=30, density=True, alpha=0.6, color='orange', label='Residuals')
    mu, sigma = stats.norm.fit(residuals)
    xmin, xmax = ax_hist.get_xlim()
    x_grid = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x_grid, mu, sigma)
    ax_hist.plot(x_grid, p, 'k', linewidth=2, label='Normal Fit')
    ax_hist.set_title(f"Residual Distribution (Horizon {horizon})", fontsize=10)
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'fanchart_horizon_{horizon}.png')
    plt.show()


if __name__ == "__main__":
   

    prices = np.loadtxt("prices.csv", delimiter=",")
  
    # 任务 1: 预测未来 1 天 
    results_1d = []
    print("\n=== 开始任务 1: 预测未来 1 天 ===")
    results_1d.append(train_ensemble_case("Stocks 1-6 -> Stock 7", prices, list(range(6)), horizon=1))
    results_1d.append(train_ensemble_case("Stock 7 -> Stock 7", prices, [6], horizon=1))
    results_1d.append(train_ensemble_case("All 7 -> Stock 7", prices, list(range(7)), horizon=1))
    
    plot_overview(results_1d, horizon=1)
    plot_fan_chart(results_1d[1], horizon=1) # 画 Case 2 的详情

    # 任务 2: 预测未来 3 天 
    results_3d = []
    print("\n=== 开始任务 2: 预测未来 3 天 ===")
    results_3d.append(train_ensemble_case("Stocks 1-6 -> Stock 7", prices, list(range(6)), horizon=3))
    results_3d.append(train_ensemble_case("Stock 7 -> Stock 7", prices, [6], horizon=3))
    results_3d.append(train_ensemble_case("All 7 -> Stock 7", prices, list(range(7)), horizon=3))
    
    plot_overview(results_3d, horizon=3)
    plot_fan_chart(results_3d[1], horizon=3) # 画 Case 2 的详情
    
    print("\n所有任务完成，图片已保存。")