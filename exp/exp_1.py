import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =========================
# 1. 数据准备
# =========================

def prepare_data(file_path="dataset/New Dow Jones.csv", 
                seq_len=20, 
                train_ratio=0.7,
                val_ratio=0.15):
    """准备训练、验证和测试数据"""
    
    # 读取数据
    df = pd.read_csv(file_path)
    prices = df['Close'].values.reshape(-1, 1)
    
    # 归一化
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)
    
    # 划分数据集
    n = len(prices_scaled)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = prices_scaled[:train_size]
    val_data = prices_scaled[train_size - seq_len:train_size + val_size]
    test_data = prices_scaled[train_size + val_size - seq_len:]
    
    # 创建数据集
    from models.model import StockDataset
    
    train_dataset = StockDataset(train_data, seq_len=seq_len)
    val_dataset = StockDataset(val_data, seq_len=seq_len)
    test_dataset = StockDataset(test_data, seq_len=seq_len)
    
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler


# =========================
# 2. 模型训练
# =========================

def train_model(model, train_loader, val_loader, 
                epochs=50, lr=0.001, device='cpu',
                early_stop_patience=10):
    """训练RNN模型"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("开始训练...")
    print("="*60)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_x).squeeze(-1)
            loss = criterion(outputs, batch_y.squeeze(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x).squeeze(-1)
                loss = criterion(outputs, batch_y.squeeze(-1))
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        # 打印信息
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Best Val: {best_val_loss:.6f}")
        
        # 早停
        if patience_counter >= early_stop_patience:
            print(f"\n早停触发! 在epoch {epoch+1}")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    print("="*60)
    print(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")
    
    return model, train_losses, val_losses


# =========================
# 3. 辅助函数：安全类型转换
# =========================

def safe_float_convert(x):
    """安全地将各种类型转换为float"""
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return 0.0
        if x.ndim > 1:
            return float(x.flatten().mean())
        elif x.size > 1:
            return float(x.mean())
        else:
            return float(x.item())
    return float(x)


# =========================
# 4. 金融指标计算
# =========================

def calculate_max_drawdown(net_value):
    """计算最大回撤"""
    try:
        # 转换为numpy数组
        if isinstance(net_value, pd.Series):
            net_value = net_value.values
        net_value = np.asarray(net_value).flatten()
        
        # 检查有效性
        if len(net_value) == 0 or not np.all(np.isfinite(net_value)):
            return 0.0
        
        # 计算累积最大值
        cum_max = np.maximum.accumulate(net_value)
        # 计算回撤比例
        drawdown = (cum_max - net_value) / (cum_max + 1e-10)
        
        return float(np.max(drawdown))
    except Exception as e:
        print(f"计算最大回撤时出错: {e}")
        return 0.0


def calculate_sharpe_ratio(returns, risk_free_rate=0.03, periods_per_year=252):
    """计算夏普比率"""
    try:
        # 转换为numpy数组
        if isinstance(returns, pd.Series):
            returns = returns.values
        returns = np.asarray(returns).flatten()
        
        # 检查有效性
        if len(returns) == 0 or not np.all(np.isfinite(returns)):
            return 0.0
        
        # 计算超额收益
        excess_return = returns - risk_free_rate / periods_per_year
        
        # 避免除零
        std_dev = excess_return.std()
        if std_dev == 0 or not np.isfinite(std_dev):
            return 0.0
        
        # 年化夏普比率
        sharpe = (excess_return.mean() * periods_per_year) / std_dev
        
        return float(sharpe)
    except Exception as e:
        print(f"计算夏普比率时出错: {e}")
        return 0.0


def calculate_win_rate(returns):
    """计算胜率"""
    try:
        if isinstance(returns, pd.Series):
            returns = returns.values
        returns = np.asarray(returns).flatten()
        
        if len(returns) == 0:
            return 0.0
        
        win_count = np.sum(returns > 0)
        return float(win_count / len(returns))
    except Exception as e:
        print(f"计算胜率时出错: {e}")
        return 0.0


def calculate_profit_loss_ratio(returns):
    """计算盈亏比"""
    try:
        if isinstance(returns, pd.Series):
            returns = returns.values
        returns = np.asarray(returns).flatten()
        
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(profits) == 0 or len(losses) == 0:
            return 0.0
        
        avg_profit = profits.mean()
        avg_loss = abs(losses.mean())
        
        if avg_loss == 0:
            return float('inf') if avg_profit > 0 else 0.0
        
        return float(avg_profit / avg_loss)
    except Exception as e:
        print(f"计算盈亏比时出错: {e}")
        return 0.0


# =========================
# 5. 模型预测和策略评估
# =========================

def evaluate_strategy(model, test_loader, device='cpu', 
                     prob_threshold=0.50,
                     save_path="results/strategy_results.csv"):
    """评估交易策略并计算金融指标"""
    
    model.eval()
    
    pred_prices = []
    true_prices = []
    prev_prices = []
    
    # 预测
    print("\n开始预测...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # 预测
            output = model(batch_x).squeeze(-1)
            
            pred_prices.append(output.cpu().numpy())
            true_prices.append(batch_y.cpu().numpy())
            prev_prices.append(batch_x[:, -1, 0].cpu().numpy())
    
    # 转换为数组
    pred_prices = np.concatenate(pred_prices)
    true_prices = np.concatenate(true_prices)
    prev_prices = np.concatenate(prev_prices)
    
    # 计算收益率
    pred_returns = (pred_prices - prev_prices) / (prev_prices + 1e-8)
    true_returns = (true_prices - prev_prices) / (prev_prices + 1e-8)
    
    # 策略回测
    print("执行策略回测...")
    records = []
    net_value = 1.0
    
    for t in range(len(pred_returns)):

        # sigmoid 映射到概率空间
        prob_up = 1 / (1 + np.exp(-pred_returns[t] * 30))
        
        if prob_up > prob_threshold:
            position = 1
        elif prob_up < (1 - prob_threshold):
            position = -1
        else:
            position = 0.2   # 轻仓

        # 策略收益（使用安全转换）
        true_ret_value = safe_float_convert(true_returns[t])
        strat_ret = position * true_ret_value
        strat_ret = np.clip(strat_ret, -0.05, 0.05)
        net_value = float(net_value * (1 + strat_ret))
        
        records.append({
            "t": t,
            "pred_price": safe_float_convert(pred_prices[t]),
            "true_price": safe_float_convert(true_prices[t]),
            "prev_price": safe_float_convert(prev_prices[t]),
            "pred_return": safe_float_convert(pred_returns[t]),
            "true_return": safe_float_convert(true_returns[t]),
            "prob_up": float(prob_up),
            "position": int(position),
            "strategy_return": float(strat_ret),
            "net_value": float(net_value)
        })
    
    df = pd.DataFrame(records)
    
    # 计算所有金融指标
    print("计算金融指标...")
    strategy_returns = df["strategy_return"].values
    net_values = df["net_value"].values
    
    # 计算各项指标
    prob_up_avg = df["prob_up"].mean()
    prob_down_avg = 1 - prob_up_avg
    max_dd = calculate_max_drawdown(net_values)
    sharpe = calculate_sharpe_ratio(strategy_returns)
    win_rate = calculate_win_rate(strategy_returns)
    pl_ratio = calculate_profit_loss_ratio(strategy_returns)
    final_value = net_values[-1] if len(net_values) > 0 else 1.0
    total_return = (final_value - 1.0) * 100
    
    # 构建指标字典（使用统一的小写加下划线命名）
    metrics = {
        "prob_up": prob_up_avg,
        "prob_down": prob_down_avg,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "profit_loss_ratio": pl_ratio,
        "final_net_value": final_value,
        "total_return_pct": total_return,
        "total_trades": len(strategy_returns)
    }
    
    # 打印指标
    print(f"\n{'='*50}")
    print(f"{'策略评估指标':^50}")
    print(f"{'='*50}")
    print(f"\n收益指标:")
    print(f"  最终净值:         {final_value:.4f}")
    print(f"  累计收益率:       {total_return:.2f}%")
    print(f"\n概率指标:")
    print(f"  平均上涨概率:     {prob_up_avg:.4f}")
    print(f"  平均下跌概率:     {prob_down_avg:.4f}")
    print(f"\n风险指标:")
    print(f"  最大回撤:         {max_dd:.4f} ({max_dd*100:.2f}%)")
    print(f"  夏普比率:         {sharpe:.4f}")
    print(f"\n交易统计:")
    print(f"  胜率:             {win_rate:.4f} ({win_rate*100:.2f}%)")
    print(f"  盈亏比:           {pl_ratio:.4f}")
    print(f"  总交易次数:       {len(strategy_returns)}")
    print(f"{'='*50}\n")
    
    # 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"结果已保存到: {save_path}")
    
    return metrics, df


# =========================
# 6. 可视化
# =========================

def plot_results(df, metrics, save_path="results/strategy_plot.png"):
    """绘制策略结果"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. 价格预测
    axes[0, 0].plot(df['true_price'], label='True Price', alpha=0.7)
    axes[0, 0].plot(df['pred_price'], label='Predicted Price', alpha=0.7)
    axes[0, 0].set_title('Price Prediction')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Normalized Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 收益率对比
    axes[0, 1].plot(df['true_return'], label='True Return', alpha=0.7)
    axes[0, 1].plot(df['pred_return'], label='Predicted Return', alpha=0.7)
    axes[0, 1].set_title('Return Prediction')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Return')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # 3. 上涨概率
    axes[1, 0].plot(df['prob_up'], label='Probability Up', color='purple')
    axes[1, 0].axhline(y=0.55, color='r', linestyle='--', 
                       label='Threshold', alpha=0.5)
    axes[1, 0].set_title('Up Probability')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 持仓信号
    axes[1, 1].plot(df['position'], label='Position', color='orange')
    axes[1, 1].set_title('Trading Position')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Position (0/1)')
    axes[1, 1].set_ylim([-0.1, 1.1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. 净值曲线（使用统一的键名）
    final_value = metrics["final_net_value"]
    axes[2, 0].plot(df['net_value'], label='Net Value', color='green', linewidth=2)
    axes[2, 0].set_title(f'Net Value Curve (Final: {final_value:.4f})')
    axes[2, 0].set_xlabel('Time')
    axes[2, 0].set_ylabel('Net Value')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. 策略收益分布（使用统一的键名）
    sharpe = metrics["sharpe_ratio"]
    axes[2, 1].hist(df['strategy_return'], bins=50, alpha=0.7, color='blue')
    axes[2, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[2, 1].set_title(f'Return Distribution (Sharpe: {sharpe:.4f})')
    axes[2, 1].set_xlabel('Return')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    plt.close()


# =========================
# 7. 主函数
# =========================

def main():
    """完整实验流程"""
    
    print("="*60)
    print("RNN股票预测与交易策略回测系统")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(2025)
    np.random.seed(2025)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 1. 准备数据
    print("\n[1/5] 准备数据...")
    try:
        train_loader, val_loader, test_loader, scaler = prepare_data(
            file_path="dataset/New Dow Jones.csv",
            seq_len=20,
            train_ratio=0.7,
            val_ratio=0.15
        )
        print(f"  训练集: {len(train_loader.dataset)} 样本")
        print(f"  验证集: {len(val_loader.dataset)} 样本")
        print(f"  测试集: {len(test_loader.dataset)} 样本")
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 创建模型
    print("\n[2/5] 创建模型...")
    try:
        from models.model import ParallelRNNModel
        
        model = ParallelRNNModel(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            fusion_method="attention"
        )
        print(f"  模型类型: ParallelRNN")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 训练模型
    print("\n[3/5] 训练模型...")
    try:
        model, train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,
            lr=0.001,
            device=device,
            early_stop_patience=15
        )
    except Exception as e:
        print(f"模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 评估策略
    print("\n[4/5] 评估交易策略...")
    try:
        metrics, df = evaluate_strategy(
            model=model,
            test_loader=test_loader,
            device=device,
            prob_threshold=0.55,
            save_path="results/parallel_rnn_strategy.csv"
        )
    except Exception as e:
        print(f"策略评估失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 可视化
    print("\n[5/5] 生成可视化...")
    try:
        plot_results(df, metrics, save_path="results/parallel_rnn_plot.png")
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)


if __name__ == "__main__":
    main()