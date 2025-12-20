import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from models.model import RNNModel, StockDataset, SeriesRNNModel, ParallelRNNModel
import time

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            if self.verbose:
                print(f'初始验证损失: {val_loss:.6f}')
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('触发早停机制!')
        else:
            if self.verbose:
                print(f'验证损失改善: {self.best_loss:.6f} -> {val_loss:.6f}')
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

def calculate_metrics(model, data_loader, device, scaler):
    """计算MSE、MAE和RMSE指标(在原始尺度上)"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算指标
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    
    # 计算MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((all_targets - all_preds) / all_targets)) * 100
    
    return mse, mae, rmse, mape, all_preds, all_targets

def calculate_baseline_metrics(test_targets_original):
    """
    计算基线模型的性能 (使用简单的持久性模型)
    持久性模型: 预测值 = 前一个时间步的真实值
    """
    # 简单预测: 用前一个值作为预测
    baseline_preds = test_targets_original[:-1]
    baseline_targets = test_targets_original[1:]
    
    baseline_mse = mean_squared_error(baseline_targets, baseline_preds)
    baseline_mae = mean_absolute_error(baseline_targets, baseline_preds)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_mape = np.mean(np.abs((baseline_targets - baseline_preds) / baseline_targets)) * 100
    
    return baseline_mse, baseline_mae, baseline_rmse, baseline_mape

def train_and_evaluate(model, model_name, train_loader, val_loader, test_loader, 
                       device, scaler, epochs=100, lr=0.001, patience=5):
    """训练并评估单个模型"""
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"{'='*60}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4, verbose=True)
    
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # 早停检查
        early_stopping(avg_val_loss, model)
        
        if early_stopping.early_stop:
            print(f"在第 {epoch+1} 轮触发早停")
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    training_time = time.time() - start_time
    
    # 在测试集上评估(归一化损失)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    
    # 计算MSE、MAE、RMSE和MAPE(原始尺度)
    test_mse, test_mae, test_rmse, test_mape, preds, targets = calculate_metrics(
        model, test_loader, device, scaler
    )
    
    # 保存模型
    model_path = f'models/{model_name}_best.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    print(f"\n{model_name} 训练完成!")
    print(f"最佳验证损失(归一化): {early_stopping.best_loss:.6f}")
    print(f"测试集损失(归一化): {avg_test_loss:.6f}")
    print(f"测试集MSE(原始尺度): {test_mse:.2f}")
    print(f"测试集RMSE(原始尺度): {test_rmse:.2f}")
    print(f"测试集MAE(原始尺度): {test_mae:.2f}")
    print(f"测试集MAPE: {test_mape:.2f}%")
    print(f"训练时间: {training_time:.2f}秒")
    print(f"模型已保存至: {model_path}")
    
    return {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': early_stopping.best_loss,
        'test_loss': avg_test_loss,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'training_time': training_time,
        'num_epochs': len(train_losses),
        'predictions': preds,
        'targets': targets
    }

# 主程序
if __name__ == "__main__":
    # 读取数据
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'dataset', 'SSE 50.csv')
    df = pd.read_csv(data_path)
    close = df["Close"].values.reshape(-1, 1)
    
    print(f"原始数据统计:")
    print(f"数据点数: {len(close)}")
    print(f"最小值: {close.min():.2f}")
    print(f"最大值: {close.max():.2f}")
    print(f"均值: {close.mean():.2f}")
    print(f"标准差: {close.std():.2f}")
    
    torch.manual_seed(2025)
    np.random.seed(2025)
    
    # 划分数据集 (70% 训练, 10% 验证, 20% 测试)
    train_size = int(len(close) * 0.7)
    val_size = int(len(close) * 0.1)
    
    train_close = close[:train_size]
    val_close = close[train_size:train_size + val_size]
    test_close = close[train_size + val_size:]
    
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_close)  # 只在训练集上fit
    val_data = scaler.transform(val_close)          # 用训练集的scaler transform
    test_data = scaler.transform(test_close)        # 用训练集的scaler transform
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_data)} ({len(train_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
    print(f"验证集: {len(val_data)} ({len(val_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
    print(f"测试集: {len(test_data)} ({len(test_data)/(len(train_data)+len(val_data)+len(test_data))*100:.1f}%)")
    
    print(f"\n归一化后的数据范围:")
    print(f"训练集: [{train_data.min():.4f}, {train_data.max():.4f}]")
    print(f"验证集: [{val_data.min():.4f}, {val_data.max():.4f}]")
    print(f"测试集: [{test_data.min():.4f}, {test_data.max():.4f}]")
    
    seq_len = 20  # 减少到20,避免模型只需要记住最后一个值
    
    # 创建数据集和数据加载器
    train_dataset = StockDataset(train_data, seq_len)
    val_dataset = StockDataset(val_data, seq_len)
    test_dataset = StockDataset(test_data, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    print(f"序列长度: {seq_len}")
    
    # 计算基线模型性能
    print(f"\n{'='*60}")
    print("基线模型 (持久性模型) 性能:")
    print(f"{'='*60}")
    baseline_mse, baseline_mae, baseline_rmse, baseline_mape = calculate_baseline_metrics(test_close)
    print(f"基线MSE: {baseline_mse:.2f}")
    print(f"基线RMSE: {baseline_rmse:.2f}")
    print(f"基线MAE: {baseline_mae:.2f}")
    print(f"基线MAPE: {baseline_mape:.2f}%")
    print(f"{'='*60}")
    
    # 定义要实验的模型
    models_to_test = [
        ("RNN", RNNModel(rnn_type="RNN").to(device)),
        ("LSTM", RNNModel(rnn_type="LSTM").to(device)),
        ("GRU", RNNModel(rnn_type="GRU").to(device)),
        ("Series_RNN", SeriesRNNModel(hidden_size=64, num_layers=2).to(device)),
        ("Parallel_RNN_Concat", ParallelRNNModel(hidden_size=64, num_layers=2, fusion_method="concat").to(device)),
        ("Parallel_RNN_Attention", ParallelRNNModel(hidden_size=64, num_layers=2, fusion_method="attention").to(device)),
    ]
    
    # 训练所有模型并收集结果
    results = []
    for model_name, model in models_to_test:
        result = train_and_evaluate(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            scaler=scaler,
            epochs=100,
            lr=0.001,
            patience=20
        )
        results.append(result)
    
    # 保存结果到CSV
    results_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Best Val Loss': r['best_val_loss'],
            'Test Loss': r['test_loss'],
            'Test MSE': r['test_mse'],
            'Test RMSE': r['test_rmse'],
            'Test MAE': r['test_mae'],
            'Test MAPE (%)': r['test_mape'],
            'Training Time (s)': r['training_time'],
            'Num Epochs': r['num_epochs'],
            'MSE vs Baseline': f"{(r['test_mse']/baseline_mse):.2%}",
            'MAE vs Baseline': f"{(r['test_mae']/baseline_mae):.2%}"
        }
        for r in results
    ])
    results_df.to_csv('experiment_results_fixed.csv', index=False)
    
    print(f"\n{'='*60}")
    print("实验结果汇总 (修复数据泄露后):")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    print(f"\n结果已保存至: experiment_results_fixed.csv")
    
    # 绘制对比图
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 训练损失对比
    ax1 = fig.add_subplot(gs[0, 0])
    for r in results:
        ax1.plot(r['train_losses'], label=r['model_name'], alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 验证损失对比
    ax2 = fig.add_subplot(gs[0, 1])
    for r in results:
        ax2.plot(r['val_losses'], label=r['model_name'], alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    # 3. MSE对比 (包含基线)
    ax3 = fig.add_subplot(gs[0, 2])
    model_names = ['Baseline'] + [r['model_name'] for r in results]
    test_mses = [baseline_mse] + [r['test_mse'] for r in results]
    colors = ['red'] + ['steelblue']*len(results)
    bars = ax3.bar(range(len(model_names)), test_mses, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.set_ylabel('MSE (Original Scale)')
    ax3.set_title('Test MSE Comparison (with Baseline)')
    ax3.grid(True, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 4. RMSE对比
    ax4 = fig.add_subplot(gs[1, 0])
    test_rmses = [baseline_rmse] + [r['test_rmse'] for r in results]
    bars = ax4.bar(range(len(model_names)), test_rmses, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=45, ha='right')
    ax4.set_ylabel('RMSE (Original Scale)')
    ax4.set_title('Test RMSE Comparison (with Baseline)')
    ax4.grid(True, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 5. MAE对比
    ax5 = fig.add_subplot(gs[1, 1])
    test_maes = [baseline_mae] + [r['test_mae'] for r in results]
    bars = ax5.bar(range(len(model_names)), test_maes, color=colors, alpha=0.7)
    ax5.set_xticks(range(len(model_names)))
    ax5.set_xticklabels(model_names, rotation=45, ha='right')
    ax5.set_ylabel('MAE (Original Scale)')
    ax5.set_title('Test MAE Comparison (with Baseline)')
    ax5.grid(True, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 6. MAPE对比
    ax6 = fig.add_subplot(gs[1, 2])
    test_mapes = [baseline_mape] + [r['test_mape'] for r in results]
    bars = ax6.bar(range(len(model_names)), test_mapes, color=colors, alpha=0.7)
    ax6.set_xticks(range(len(model_names)))
    ax6.set_xticklabels(model_names, rotation=45, ha='right')
    ax6.set_ylabel('MAPE (%)')
    ax6.set_title('Test MAPE Comparison (with Baseline)')
    ax6.grid(True, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # 7. 预测vs真实值 (最佳模型)
    best_model_idx = np.argmin([r['test_mse'] for r in results])
    best_result = results[best_model_idx]
    
    ax7 = fig.add_subplot(gs[2, :2])
    sample_size = min(200, len(best_result['targets']))
    ax7.plot(best_result['targets'][:sample_size], label='True Values', alpha=0.7, linewidth=2)
    ax7.plot(best_result['predictions'][:sample_size], label='Predictions', alpha=0.7, linewidth=2)
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('Stock Price')
    ax7.set_title(f'Predictions vs True Values ({best_result["model_name"]})')
    ax7.legend()
    ax7.grid(True)
    
    # 8. 训练时间对比
    ax8 = fig.add_subplot(gs[2, 2])
    training_times = [r['training_time'] for r in results]
    bars = ax8.bar(range(len(results)), training_times, color='plum', alpha=0.7)
    ax8.set_xticks(range(len(results)))
    ax8.set_xticklabels([r['model_name'] for r in results], rotation=45, ha='right')
    ax8.set_ylabel('Training Time (seconds)')
    ax8.set_title('Training Time Comparison')
    ax8.grid(True, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
    
    plt.savefig('experiment_comparison_fixed.png', dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: experiment_comparison_fixed.png")
    plt.show()
    
    # 找出最佳模型
    best_model = results[best_model_idx]
    print(f"\n{'='*60}")
    print(f"最佳模型: {best_model['model_name']}")
    print(f"测试集MSE: {best_model['test_mse']:.2f} (基线: {baseline_mse:.2f}, 改进: {(1-best_model['test_mse']/baseline_mse)*100:.1f}%)")
    print(f"测试集RMSE: {best_model['test_rmse']:.2f} (基线: {baseline_rmse:.2f})")
    print(f"测试集MAE: {best_model['test_mae']:.2f} (基线: {baseline_mae:.2f}, 改进: {(1-best_model['test_mae']/baseline_mae)*100:.1f}%)")
    print(f"测试集MAPE: {best_model['test_mape']:.2f}% (基线: {baseline_mape:.2f}%)")
    print(f"训练时间: {best_model['training_time']:.2f}秒")

    print(f"{'='*60}")
