from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from ai_quant.config import load_config
from ai_quant.data import load_market_csv, load_holdings_csv, fetch_akshare_history
from ai_quant.engine import score_universe
from ai_quant.features import add_features
from ai_quant.ml_model import train_model
from ai_quant.backtest import backtest_signals
from ai_quant.report import top_report


def main():
    p = argparse.ArgumentParser(description="洪老师 AI 量化系统 v1.0")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="对最新交易日全池评分")
    s.add_argument("--data", required=True)
    s.add_argument("--holdings")
    s.add_argument("--model", default="models/model.joblib")
    s.add_argument("--out", default="output_scores.csv")
    s.add_argument("--top", type=int, default=20)

    t = sub.add_parser("train", help="训练AI预测层（严格按时间切分）")
    t.add_argument("--data", required=True)
    t.add_argument("--model", default="models/model.joblib")

    b = sub.add_parser("backtest", help="回测规则评分信号")
    b.add_argument("--data", required=True)
    b.add_argument("--out", default="backtest_trades.csv")
    b.add_argument("--score", type=float, default=None)

    d = sub.add_parser("download", help="使用AkShare下载日线")
    d.add_argument("--symbols", required=True, help="逗号分隔，如 600584,000938")
    d.add_argument("--start", required=True, help="YYYYMMDD")
    d.add_argument("--end", required=True, help="YYYYMMDD")
    d.add_argument("--out", default="market.csv")

    args = p.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "download":
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        df = fetch_akshare_history(syms, args.start, args.end, args.out)
        print(f"已保存 {len(df)} 行到 {args.out}")
        return

    market = load_market_csv(args.data)

    if args.cmd == "score":
        holdings = load_holdings_csv(args.holdings)
        model_path = args.model if Path(args.model).exists() else None
        scores, risk = score_universe(market, cfg, holdings, model_path)
        scores.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(top_report(scores, args.top))
        if risk:
            print("\n组合风险:", json.dumps(risk, ensure_ascii=False, indent=2))
        print(f"\n完整结果已保存: {args.out}")

    elif args.cmd == "train":
        feat = add_features(market)
        info = train_model(feat, cfg, args.model)
        print(info)

    elif args.cmd == "backtest":
        trades, stats = backtest_signals(market, cfg, args.score)
        trades.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"交易明细已保存: {args.out}")

if __name__ == "__main__":
    main()
