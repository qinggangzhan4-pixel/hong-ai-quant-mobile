# 洪老师 AI 量化 v2.0（手机网页版）

这是 v1.0 的移动端升级版，核心量化引擎不变，重做了 Streamlit 手机界面。

## 手机端能做什么

- 今日候选 Top N 卡片
- 输入股票代码/名称查询
- 手机直接录入持仓
- 自动检查总仓、单票、单行业集中度
- 上传行情 CSV 后一键全市场评分
- 显示规则分、AI概率、建议首仓、ATR止损
- 首板/一进二仍使用独立 relay 模块

## 一键本地启动

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 把本目录上传到 GitHub 仓库。
2. 在 Streamlit Community Cloud 选择该仓库。
3. Main file path 填 `app.py`。
4. 部署完成后会得到一个 HTTPS 网页地址。
5. iPhone / Android 浏览器打开即可；可“添加到主屏幕”。

## 数据要求

最少字段：

`date,symbol,open,high,low,close,volume`

推荐补充：

`name,industry,amount,turnover_rate,market_cap,pe_ttm,pb,roe,net_profit_yoy,main_net_inflow,auction_open_ratio,auction_volume_ratio,is_limit_up,is_st`

数据越完整，行业/质量/资金/接力评分越有意义。系统不会伪造缺失字段。

## 重要限制

当前网页本身不含券商交易接口，也不自动抓取同花顺私有盘口。它是“量化决策终端”，不是自动下单机器人。
