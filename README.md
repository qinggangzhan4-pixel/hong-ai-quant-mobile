# 洪老师 AI 量化系统 v1.0

这是基于你提供的 802 份量化资料 + 《A股中线策略体系 v6.2》整理出的 **可运行 MVP**。

它不是“AI猜明天涨停”，而是：

**市场状态 → 行业 → 个股质量 → 趋势量价 → 首板接力 → AI概率 → 仓位/止损**

其中 AI 只有在样本外测试达标后才参与评分；不达标就自动关闭。

## 核心特征

- 默认排除科创板、创业板、北交所（可改 config.yaml）
- 100分规则评分
  - 市场环境 20
  - 行业强度 20
  - 个股质量 20
  - 趋势量价 25
  - 执行质量 15
- 首板/一进二专用模块
- 市场宽度（Close > MA20 比例）
- ATR动态止损
- 单笔风险预算倒推仓位
- 单票 / 单行业 / 总仓风险控制
- AI层使用 HistGradientBoosting，严格按日期切分测试
- AI样本外 AUC < 0.53 自动禁用
- 回测遵守“信号日收盘 → 次日开盘买入”，并考虑 A 股 T+1

## 1. 安装

```bash
cd hong_ai_quant_v1
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 准备数据

最少字段：

```text
date,symbol,open,high,low,close,volume
```

可选但建议：

```text
name,industry,amount,turnover_rate,market_cap,pe_ttm,pb,roe,net_profit_yoy,
main_net_inflow,auction_open_ratio,auction_volume_ratio,is_limit_up,is_st
```

### 用 AkShare 下载基础日线

```bash
python run.py download --symbols 600584,000938,000977 --start 20240101 --end 20261119 --out market.csv
```

AkShare只负责基础日线；行业、基本面、资金、竞价字段建议从你的行情源补齐，字段缺失时系统会降级运行，不会伪造。

## 3. 评分

```bash
python run.py score --data market.csv --out scores.csv
```

如果有当前持仓：

```bash
python run.py score --data market.csv --holdings examples/holdings_template.csv --out scores.csv
```

持仓格式：

```text
symbol,name,industry,market_value
600901,江苏金租,非银金融,24896
600584,长电科技,半导体,5547
```

系统会把单行业集中度和单票集中度计入风险提示。

## 4. 训练AI层

先准备至少数百到上千条历史样本：

```bash
python run.py train --data market.csv --model models/model.joblib
```

模型目标默认定义为：**未来3个交易日内最高价相对信号日收盘上涨至少3%**。

重要：
- 训练/测试严格按日期切开；
- 若样本外 AUC < 0.53，模型保存但 `enabled=false`，正式评分不会使用它；
- 这避免“AI在历史上看起来很聪明，实盘却完全失效”。

## 5. 回测

```bash
python run.py backtest --data market.csv --out backtest_trades.csv
```

规则：
- t日收盘出信号；
- t+1开盘成交；
- 买入当天不卖（T+1）；
- 后续按ATR止损 / +8%第一止盈 / 5日时间止损；
- 扣除滑点、佣金、印花税近似值。

## 6. 可视化界面

```bash
streamlit run app.py
```

浏览器里直接上传 CSV，就能得到排名和组合风险。

## 7. 默认仓位纪律

当前 config.yaml 默认针对你约4.5万元账户：

- 普通首仓：6%
- 单票上限：15%
- 单行业目标上限：30%
- 单笔风险：0.7%
- 正常总仓：70%
- 弱市总仓：45%
- 强市最高：80%
- 短线止损：ATR×2，限制在3%~6%
- 时间止损：5个交易日

## 8. 评分怎么理解

- **>=75**：允许首仓候选，但仍必须看组合风险
- **60~74**：观察，等待回踩/突破确认
- **<60**：不主动开仓

涨停/首板票另走 `relay_score`：
- >=70：接力候选
- 50~69：观察
- <50：回避

## 9. 这版还没有做什么

v1.0 故意不做：
- 自动连接券商直接下单
- 盘中毫秒级盘口
- 自动抓取新闻情绪
- 自动读取同花顺私有资金流
- “明天必涨”式黑箱预测

原因是先把**可验证、可回测、可复盘**的底层跑通。

下一版最值得加的是：
1. 你的 NeoData / 同花顺数据导入适配器；
2. 盘前 9:25 竞价扫描；
3. 纳指/费半/日韩作为开盘情绪因子；
4. 每天自动输出“持仓减仓/加仓/新候选”三张表；
5. 交易日志自动学习你的真实结果。

## 10. 免责声明

该系统用于研究和决策辅助，不保证收益。AI概率是历史样本统计，不等于未来确定性。仓位与止损规则优先级高于AI分数。
