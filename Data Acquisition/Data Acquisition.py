import requests
import re
import json
import pandas as pd
import time
import akshare as ak  # 仅用于道琼斯


# 从新浪财经爬取上证指数
def fetch_sse_index():
    print("正在从【新浪财经】获取上证指数...")
    url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_sh000001=/CN_MarketDataService.getKLineData"
    params = {"symbol": "sh000001", "scale": "240", "ma": "no", "datalen": "1023"}
    try:
        response = requests.get(url, params=params, timeout=10)
        match = re.search(r'\((.*)\)', response.text)
        if not match:
            raise ValueError("无法解析 JSONP 数据")
        data = json.loads(match.group(1))

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['day'])
        df.set_index('date', inplace=True)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'      # 注意！！！！！我代码这里的Volume 实际是 成交额（元），不是成交量（手）。这是新浪接口的特点。
        }, inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        print(f"上证指数获取成功，共 {len(df)} 条记录")
        return df.sort_index()
    except Exception as e:
        print(f"上证指数获取失败: {e}")
        return None



# 道琼斯指数：来自【AKShare】（使用 sina 源，稳定可靠）

"""
别的网站都有反爬机制
目前这个网站可以爬到
"""

def fetch_dow_jones_akshare():
    print("正在通过【AKShare】获取道琼斯指数（来源：Sina Finance）...")
    try:
        # AKShare 内置接口，实际数据源是新浪财经美股板块，但封装好了
        df = ak.index_us_stock_sina(symbol="DJIA")
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        print(f"道琼斯指数获取成功，共 {len(df)} 条记录")
        return df.sort_index()
    except Exception as e:
        print(f"道琼斯指数获取失败: {e}")
        return None


# ==============================
# 主程序
# ==============================
if __name__ == "__main__":
    print("开始抓取金融指数数据（上证：新浪财经 | 道琼斯：AKShare）...\n")

    # 获取上证
    sse_df = fetch_sse_index()
    time.sleep(1)  # 稍作延迟

    # 获取道琼斯
    dow_df = fetch_dow_jones_akshare()

    # 保存文件
    saved = False
    if sse_df is not None:
        try:
            sse_df.to_csv("上证指数_新浪财经.csv", encoding='utf-8-sig')
            print("上证指数已保存为：上证指数_新浪财经.csv")
            saved = True
        except PermissionError:
            print("⚠️   无法保存上证指数：请关闭 Excel 或其他占用该文件的程序！")

    if dow_df is not None:
        try:
            dow_df.to_csv("道琼斯指数_AKShare.csv", encoding='utf-8-sig')
            print("道琼斯指数已保存为：道琼斯指数_AKShare.csv")
            saved = True
        except PermissionError:
            print("⚠️  无法保存道琼斯指数：请关闭 Excel 或其他占用该文件的程序！")

    # 打印预览
    print("\n" + "=" * 50)
    if sse_df is not None:
        print("\n上证指数最新5日：")
        print(sse_df.tail())
    if dow_df is not None:
        print("\n道琼斯指数最新5日：")
        print(dow_df.tail())

    if not saved:
        print("\n❗ 所有数据均未保存，请检查错误或文件占用情况。")