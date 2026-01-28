#!/usr/bin/env python3
"""
Gambler Detector — single-file Flask app (no equity curve).
- Re-introduced drawdown penalty (starts at 50% of max-loss limit, stepped).
- Shows average winning and losing trade duration (minutes).
Run:
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install flask pandas numpy
    python gambler_detector_web_app.py
Open http://127.0.0.1:5000
"""

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import re

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024

# ---------- utilities ----------
MAPS = {
    'entry': ['Open', 'Open time', 'Time', 'Entry'],
    'exit':  ['Close', 'Close time', 'Time.1', 'Exit'],
    'profit':['Profit', 'profit', 'P/L'],
    'comm':  ['Commissions', 'Commission', 'commission'],
    'swap':  ['Swap', 'swaps', 'swap'],
    'volume':['Volume', 'Lots'],
    'symbol':['Symbol', 'Item']
}
MQL5_ID_CANDS = ['Position', 'Position ID', 'Ticket', 'Order', 'Order ID', 'Deal']
MQL5_TIME_CANDS = ['Time','Deal Time','time','DealTime','Date']
MQL5_TYPE_CANDS = ['Type','Deal type','Action','Type']

def find_column(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        k = cand.lower()
        if k in cols_lower:
            return cols_lower[k]
    for cand in candidates:
        for c in columns:
            if cand.lower() in c.lower():
                return c
    return None

def clean_num(x):
    if x is None:
        return 0.0
    if isinstance(x, pd.Series):
        s = x.fillna('').astype(str)
        s = s.str.replace(r'[\$\s\u00A0]', '', regex=True)
        s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
        s = s.str.replace(',', '', regex=False)
        s = s.str.replace(r'[^0-9\.\-]', '', regex=True)
        return pd.to_numeric(s, errors='coerce').fillna(0.0)
    else:
        if pd.isna(x):
            return 0.0
        s = str(x)
        s = re.sub(r'[\$\s\u00A0]', '', s)
        if s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
        s = s.replace(',', '')
        s = re.sub(r'[^0-9\.\-]', '', s)
        try:
            return float(s)
        except:
            return 0.0

def parse_dt_safe(df, col):
    if col and col in df.columns:
        return pd.to_datetime(df[col], errors='coerce')
    else:
        return pd.Series([pd.NaT]*len(df), index=df.index)

def fmt_currency(x):
    try: return f"{x:,.2f}"
    except: return str(x)
def fmt_num(x, prec=2):
    try: return f"{x:.{prec}f}"
    except: return str(x)

# Build flagged-table that includes previous trade row for each flagged index
def build_flag_with_previous_table(df, mask, cols_to_show=None, maxrows=500):
    """
    For each True in mask, output:
      - previous trade (index-1) if exists
      - flagged trade (index)
    Concatenate blocks vertically. Avoid duplicates if consecutive flagged trades would duplicate previous rows.
    Returns HTML table (no 'role' column).
    """
    if cols_to_show is None:
        cols_to_show = ['entry_time','exit_time','symbol','volume','profit']

    flagged_idxs = [int(i) for i in df[mask].index]
    if not flagged_idxs:
        return None

    rows = []
    used = set()
    for idx in flagged_idxs:
        prev_idx = idx - 1
        if prev_idx >= 0 and prev_idx not in used:
            rows.append(prev_idx)
            used.add(prev_idx)
        if idx not in used:
            rows.append(idx)
            used.add(idx)

    # Build DataFrame from selected rows
    out_rows = [df.iloc[i:i+1].copy() for i in rows[:maxrows]]
    if not out_rows:
        return None
    out_df = pd.concat(out_rows, ignore_index=True)

    # Keep only requested columns present + format datetimes
    keep = [c for c in cols_to_show if c in out_df.columns]
    out_df = out_df[keep]
    for tcol in ['entry_time','exit_time']:
        if tcol in out_df.columns:
            out_df[tcol] = pd.to_datetime(out_df[tcol], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    return out_df.head(maxrows).to_html(index=False, classes='compact')

# ---------- core processing ----------
def process_dataframe(raw: pd.DataFrame, STARTING_BALANCE: float, MAX_LOSS_LIMIT_USD: float, DAILY_LOSS_PCT: float):
    # map columns
    cols = {k: find_column(raw.columns.tolist(), v) for k, v in MAPS.items()}
    id_col = find_column(raw.columns.tolist(), MQL5_ID_CANDS)
    time_guess = find_column(raw.columns.tolist(), MQL5_TIME_CANDS)
    type_col = find_column(raw.columns.tolist(), MQL5_TYPE_CANDS)

    profit_col = cols.get('profit')
    if profit_col is None:
        for c in raw.columns:
            if 'profit' in c.lower() or 'p/l' in c.lower() or 'p&l' in c.lower():
                profit_col = c
                break

    df_raw = raw.copy()
    df_raw['entry_time_ftmo'] = parse_dt_safe(df_raw, cols.get('entry'))
    df_raw['exit_time_ftmo']  = parse_dt_safe(df_raw, cols.get('exit'))
    ftmo_like = (df_raw['exit_time_ftmo'].notna().sum() > 0) and (df_raw['entry_time_ftmo'].notna().sum() > 0)

    if ftmo_like:
        df = df_raw.copy()
        df['entry_time'] = df['entry_time_ftmo']
        df['exit_time']  = df['exit_time_ftmo']
        df['raw_p'] = clean_num(df[profit_col]) if profit_col in df.columns else 0.0
        df['comm'] = clean_num(df[cols.get('comm')]) if cols.get('comm') in df.columns else 0.0
        df['swap'] = clean_num(df[cols.get('swap')]) if cols.get('swap') in df.columns else 0.0
        df['profit'] = df['raw_p'] + df['comm'] + df['swap']
        df['symbol'] = df[cols.get('symbol')].astype(str).str.strip() if cols.get('symbol') in df.columns else "Unknown"
        df['volume'] = clean_num(df[cols.get('volume')]) if cols.get('volume') in df.columns else 0.0
        df = df.dropna(subset=['entry_time','exit_time','profit']).sort_values('exit_time').reset_index(drop=True)
    else:
        raw2 = df_raw.copy()
        time_col = time_guess if (time_guess and time_guess in raw2.columns) else None
        if time_col is None:
            for c in raw2.columns:
                try_dt = pd.to_datetime(raw2[c], errors='coerce')
                if try_dt.notna().sum() >= max(1, len(raw2)//10):
                    time_col = c; break
        if time_col is None:
            for c in raw2.columns:
                if 'time' in c.lower():
                    time_col = c; break

        raw2['_t'] = pd.to_datetime(raw2[time_col], errors='coerce') if (time_col and time_col in raw2.columns) else pd.NaT
        raw2['_p'] = clean_num(raw2[profit_col]) if profit_col in raw2.columns else 0.0
        raw2['_c'] = clean_num(raw2[cols.get('comm')]) if cols.get('comm') in raw2.columns else 0.0
        raw2['_s'] = clean_num(raw2[cols.get('swap')]) if cols.get('swap') in raw2.columns else 0.0

        group_col = id_col if (id_col and id_col in raw2.columns) else None
        if not group_col:
            candidates = []
            exclude = {time_col, profit_col, cols.get('comm'), cols.get('swap')}
            for c in raw2.columns:
                if c in exclude: continue
                nonnull = raw2[c].dropna().astype(str)
                if len(nonnull) < 3: continue
                dupcount = len(nonnull) - nonnull.nunique()
                if dupcount > 0:
                    candidates.append((c, dupcount))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                group_col = candidates[0][0]

        trades = []
        if group_col:
            for pid, g in raw2.groupby(group_col):
                g = g.sort_values('_t')
                if g['_t'].notna().sum() == 0: continue
                entry_time = g['_t'].min()
                if type_col and type_col in g.columns:
                    types = g[type_col].astype(str).str.lower().fillna('')
                    bs = g[types.str.contains(r'\bbuy\b|\bsell\b', regex=True, na=False)]
                    if not bs.empty:
                        zero = bs[bs['_p'].abs() < 1e-9]
                        entry_time = zero['_t'].iloc[0] if not zero.empty else bs['_t'].iloc[0]
                exits = g[g['_p'].abs() > 1e-9]
                exit_time = exits['_t'].iloc[-1] if not exits.empty else g['_t'].max()
                raw_p_sum = float(g['_p'].sum()); comm_sum = float(g['_c'].sum()); swap_sum = float(g['_s'].sum())
                net = raw_p_sum + comm_sum + swap_sum
                sym = g[cols.get('symbol')].iloc[0] if (cols.get('symbol') in g.columns) else "Unknown"
                vol = clean_num(g[cols.get('volume')].iloc[0]) if (cols.get('volume') in g.columns) else 0.0
                trades.append({'entry_time': entry_time, 'exit_time': exit_time, 'raw_p': raw_p_sum,
                               'comm': comm_sum, 'swap': swap_sum, 'profit': net, 'symbol': sym, 'volume': vol})
        if not trades:
            for _, r in raw2.iterrows():
                t = r.get('_t', pd.NaT)
                raw_p = float(r.get('_p', 0.0)); commv = float(r.get('_c', 0.0)); swapv = float(r.get('_s', 0.0))
                net = raw_p + commv + swapv
                sym = r[cols.get('symbol')] if (cols.get('symbol') in raw2.columns) else "Unknown"
                vol = clean_num(r[cols.get('volume')]) if (cols.get('volume') in raw2.columns) else 0.0
                trades.append({'entry_time': t, 'exit_time': t, 'raw_p': raw_p, 'comm': commv, 'swap': swapv,
                               'profit': net, 'symbol': sym, 'volume': vol})
        df = pd.DataFrame(trades)
        for c in ['raw_p','comm','swap','profit','symbol','volume','entry_time','exit_time']:
            if c not in df.columns:
                df[c] = 0.0 if c not in ('entry_time','exit_time','symbol') else (pd.NaT if c in ('entry_time','exit_time') else "Unknown")
        df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
        df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
        df = df.dropna(subset=['exit_time', 'profit']).sort_values('exit_time').reset_index(drop=True)

    # ensure numeric cols
    if 'raw_p' not in df.columns: df['raw_p'] = df['profit'] - df.get('comm', 0.0) - df.get('swap', 0.0)
    if 'comm' not in df.columns: df['comm'] = 0.0
    if 'swap' not in df.columns: df['swap'] = 0.0
    df['profit'] = df['raw_p'] + df['comm'] + df['swap']
    if 'symbol' not in df.columns: df['symbol'] = "Unknown"
    if 'volume' not in df.columns: df['volume'] = 0.0
    df = df.dropna(subset=['entry_time', 'exit_time', 'profit']).sort_values('exit_time').reset_index(drop=True)

    # analytics (no equity curve)
    n_trades = len(df)
    total_comm = df['comm'].sum()
    total_swap = df['swap'].sum()
    net_profit = df['profit'].sum()
    wins = df[df['profit'] > 0]; losses = df[df['profit'] <= 0]
    win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0.0
    avg_win = wins['profit'].mean() if not wins.empty else 0.0
    avg_loss = losses['profit'].mean() if not losses.empty else 0.0
    profit_factor = (wins['profit'].sum() / abs(losses['profit'].sum())) if (not losses.empty and abs(losses['profit'].sum())>0) else 0.0
    avg_rrr = (avg_win / abs(avg_loss)) if (avg_loss != 0) else avg_win
    expectancy = ((win_rate/100) * avg_win) + (((100-win_rate)/100) * avg_loss)
    sharpe = (df['profit'].mean() / df['profit'].std() * np.sqrt(252)) if df['profit'].std() != 0 else 0.0

    df['duration_min'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60.0
    avg_win_hold = df[df['profit'] > 0]['duration_min'].mean()
    avg_loss_hold = df[df['profit'] <= 0]['duration_min'].mean()
    hold_ratio = (avg_loss_hold / avg_win_hold) if (avg_win_hold and avg_win_hold > 0) else 1.0
    median_win_hold = df[df['profit'] > 0]['duration_min'].median()
    median_loss_hold = df[df['profit'] <= 0]['duration_min'].median()

    df['prev_net'] = df['profit'].shift(1)
    df['prev_symbol'] = df['symbol'].shift(1)
    df['prev_exit'] = df['exit_time'].shift(1)
    df['vol_change'] = df['volume'] / df['volume'].shift(1)
    df['gap_min'] = (df['entry_time'] - df['prev_exit']).dt.total_seconds() / 60.0

    rev_mask = (df['prev_net'] < 0) & (df['symbol'] == df['prev_symbol']) & (df['gap_min'] >= 0) & (df['gap_min'] <= 5.0)
    mart_mask = (df['prev_net'] < 0) & (df['symbol'] == df['prev_symbol']) & (df['vol_change'] >= 1.8)
    df['is_hedged'] = (df['gap_min'] < 0) & (df['symbol'] == df['prev_symbol'])
    n_hedges = int(df['is_hedged'].sum())
    hedge_rate = (n_hedges / n_trades * 100) if n_trades > 0 else 0.0
    is_bot_grid = (hedge_rate > 20.0) and (n_trades >= 15)

    # equity & drawdown (for penalty)
    df['equity'] = STARTING_BALANCE + df['profit'].cumsum()
    df['peak'] = df['equity'].cummax()
    abs_drawdown_usd = STARTING_BALANCE - df['equity'].min() if df['equity'].min() < STARTING_BALANCE else 0.0
    proximity_pct = (abs_drawdown_usd / MAX_LOSS_LIMIT_USD) * 100 if MAX_LOSS_LIMIT_USD else 0.0
    max_p2v_dd = (df['peak'] - df['equity']).max() if 'peak' in df.columns else 0.0

    df['exit_date'] = df['exit_time'].dt.date
    daily_pnl = df.groupby('exit_date')['profit'].sum()
    daily_loss_threshold_usd = -(STARTING_BALANCE * (DAILY_LOSS_PCT / 100.0))
    daily_breaches = daily_pnl[daily_pnl <= daily_loss_threshold_usd]
    n_daily_breaches = len(daily_breaches)

    rr_skew = (abs(avg_loss) / avg_win) if avg_win > 0 else 0.0
    is_hidden_holder = (rr_skew > 3.0) and (hold_ratio > 2.0)

    rev_count = int(rev_mask.sum())
    rev_rate = (rev_count / n_trades * 100) if n_trades > 0 else 0.0
    mart_count = int(mart_mask.sum())

    # scoring with drawdown penalty reinstated
    score = 100
    penalties = []

    # Drawdown proximity penalty: -2 pts every 5% starting at 50% (matches previous logic)
    if proximity_pct >= 50:
        steps = int((proximity_pct - 45) / 5)
        deduction = steps * 2
        score -= deduction
        penalties.append(f"Net Drawdown Penalty ({proximity_pct:.1f}% of limit used) | -{deduction} pts")

    if rev_count > 1:
        rev_deduction = min(25, int(rev_rate * 2)); score -= rev_deduction
        penalties.append(f"Revenge Trading Habit ({rev_rate:.1f}% frequency) | -{rev_deduction} pts")
    elif rev_count == 1:
        penalties.append("Minor Execution Noise (1 Revenge-style entry) | -0 pts")

    if n_hedges > 0:
        hedge_penalty = 5 if n_hedges <= 2 else 30; score -= hedge_penalty
        penalties.append(f"Hedging/Stacking Detected ({n_hedges} times) | -{hedge_penalty} pts")

    if mart_count > 1:
        mart_deduction = min(30, mart_count * 10); score -= mart_deduction
        penalties.append(f"Martingale Patterns Detected ({mart_count} times) | -{mart_deduction} pts")
    elif mart_count == 1:
        penalties.append("Isolated Lot Size Spike (1 time) | -0 pts")

    gross_profit_wins = wins['profit'].sum() if not wins.empty else 0.0
    if gross_profit_wins > 0:
        consistency_val = (wins['profit'].max() / gross_profit_wins * 100)
        if consistency_val > 30:
            penalties.append(f"Consistency Warning ({consistency_val:.1f}%) | -0 pts (Warning Only)")

    if n_daily_breaches > 0:
        daily_deduction = n_daily_breaches * 20; score -= daily_deduction
        penalties.append(f"Daily Loss Limit Breached ({n_daily_breaches} days) | -{daily_deduction} pts")

    if is_hidden_holder:
        skew_penalty = 20; score -= skew_penalty
        penalties.append(f"Toxic Risk-Reward Skew (Hidden Holding) | -{skew_penalty} pts")

    score = max(0, score)
    rating = "INSTITUTIONAL" if score >= 94 else "PROFESSIONAL" if score >= 82 else "DEVELOPING" if score >= 55 else "GAMBLER"

    # generate small HTML tables (pandas), showing previous row then flagged row
    cols_to_show = ['entry_time','exit_time','symbol','volume','profit']
    rev_table_html = build_flag_with_previous_table(df, rev_mask, cols_to_show=cols_to_show) if rev_mask.any() else None
    mart_table_html = build_flag_with_previous_table(df, mart_mask, cols_to_show=cols_to_show) if mart_mask.any() else None
    hedge_table_html = df[df['is_hedged']][cols_to_show].head(200).copy()
    for tcol in ['entry_time','exit_time']:
        if tcol in hedge_table_html.columns:
            hedge_table_html[tcol] = pd.to_datetime(hedge_table_html[tcol], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    hedge_table_html = hedge_table_html.to_html(index=False, classes='compact') if not hedge_table_html.empty else None
    sample_html = df[cols_to_show].head(200).copy()
    for tcol in ['entry_time','exit_time']:
        if tcol in sample_html.columns:
            sample_html[tcol] = pd.to_datetime(sample_html[tcol], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    sample_html = sample_html.to_html(index=False, classes='compact')

    results = {
        'n_trades': int(n_trades),
        'total_comm_fmt': fmt_currency(total_comm),
        'total_swap_fmt': fmt_currency(total_swap),
        'net_profit_fmt': fmt_currency(net_profit),
        'win_rate_fmt': fmt_num(win_rate,2),
        'profit_factor_fmt': fmt_num(profit_factor,2),
        'expectancy_fmt': fmt_currency(expectancy),
        'sharpe_fmt': fmt_num(sharpe,2),
        'avg_win_hold': f"{avg_win_hold:.1f}" if not pd.isna(avg_win_hold) else "N/A",
        'avg_loss_hold': f"{avg_loss_hold:.1f}" if not pd.isna(avg_loss_hold) else "N/A",
        'hedge_rate_fmt': fmt_num(hedge_rate,2),
        'is_bot_grid': bool(is_bot_grid),
        'score': int(score),
        'rating': rating,
        'penalties': penalties,
        'consistency_warning': "",
        'rev_table_html': rev_table_html,
        'mart_table_html': mart_table_html,
        'hedge_table_html': hedge_table_html,
        'df_sample_html': sample_html,
        'daily_breaches': {str(k): fmt_currency(v) for k, v in (daily_breaches.items() if 'daily_breaches' in locals() else {})}
    }
    return results

# ---------- templates ----------
INDEX_HTML = """
<!doctype html>
<html><head><meta charset="utf-8"><title>Gambler Detector</title>
<style>
  body{font-family:Arial,Helvetica,sans-serif;margin:20px}
  .card{border:1px solid #ddd;padding:18px;border-radius:6px;max-width:980px}
  form label{display:block;margin-top:18px;margin-bottom:8px;font-weight:600}
  form input[type="number"], form input[type="file"]{display:block;padding:10px;margin-bottom:18px;width:100%;box-sizing:border-box}
  button{padding:12px 16px;border-radius:6px}
</style>
</head><body>
  <h2>Gambler Detector</h2>
  <div class="card">
    <form action="/analyze" method="post" enctype="multipart/form-data">
      <label>Starting Balance</label>
      <input name="starting_balance" type="number" step="1" value="200000" required>
      <label>Max Loss Limit (USD)</label>
      <input name="max_loss_limit_usd" type="number" step="1" value="20000" required>
      <label>Daily Loss %</label>
      <input name="daily_loss_pct" type="number" step="0.1" value="5" required>
      <label>History CSV</label>
      <input name="history_csv" type="file" accept=".csv" required>
      <br>
      <button type="submit">Analyze</button>
    </form>
  </div>
</body></html>
"""

RESULT_HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>Results</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;margin:20px}
.container{max-width:1200px;margin:0 auto}
.card{border:1px solid #ddd;padding:12px;margin:12px 0;border-radius:6px;background:#fafafa}
.table-container{max-height:320px;overflow:auto;border:1px solid #eee;padding:6px;background:#fff}
.compact table{border-collapse:collapse;width:100%;font-size:13px}
.compact th{position:sticky;top:0;background:#f3f3f3}
.compact td,.compact th{padding:6px;border:1px solid #ddd;white-space:nowrap}
</style>
</head><body>
  <div class="container">
    <h2>Analysis Results</h2>
<p style="font-size:18px;margin:4px 0;">
  <strong>Discipline Score:</strong> {{score}} / 100
</p>
<p style="font-size:18px;margin:4px 0;">
  <strong>Rating:</strong> {{rating}}
</p>

    <div class="card">
      <h3>Summary</h3>
      <pre>
Total Trades: {{n_trades}}
Net Profit/Loss: ${{net_profit_fmt}}
Win Rate: {{win_rate_fmt}}%
Profit Factor: {{profit_factor_fmt}}
Expectancy (Per Trade): ${{expectancy_fmt}}
Sharpe Ratio: {{sharpe_fmt}}
Hedge Rate: {{hedge_rate_fmt}}%

Average Winning Trade Duration: {{avg_win_hold}} min
Average Losing Trade Duration:  {{avg_loss_hold}} min
      </pre>
    </div>

    <div class="card">
      <h3>Flags & Penalties</h3>
      <ul>{% for p in penalties %}<li>{{p}}</li>{% endfor %}</ul>
      {% if consistency_warning %}<p style="color:darkorange;font-weight:700">{{consistency_warning}}</p>{% endif %}
    </div>

    <div class="card">
      <h3>Flagged Trades</h3>
      {% if rev_table_html %}<h4>Revenge</h4><div class="table-container compact">{{rev_table_html|safe}}</div>{% endif %}
      {% if mart_table_html %}<h4>Martingale</h4><div class="table-container compact">{{mart_table_html|safe}}</div>{% endif %}
      {% if hedge_table_html %}<h4>Hedging</h4><div class="table-container compact">{{hedge_table_html|safe}}</div>{% endif %}
      {% if not rev_table_html and not mart_table_html and not hedge_table_html %}<p>No flagged trades found.</p>{% endif %}
    </div>

    <div class="card">
      <h3>Trade Sample</h3>
      <div class="table-container compact">{{df_sample_html|safe}}</div>
    </div>

    <a href="/">← Back</a>
  </div>
</body></html>
"""

# ---------- routes ----------
@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        starting_balance = float(request.form.get('starting_balance', 200000))
        max_loss_limit_usd = float(request.form.get('max_loss_limit_usd', 20000))
        daily_loss_pct = float(request.form.get('daily_loss_pct', 5))
    except Exception as e:
        return f"Invalid numeric inputs: {e}", 400

    f = request.files.get('history_csv')
    if not f:
        return "No file uploaded", 400

    try:
        raw = pd.read_csv(f, sep=None, engine='python', dtype=str, encoding='utf-8-sig')
    except Exception as e:
        return f"Failed to read CSV: {e}", 400

    try:
        results = process_dataframe(raw, starting_balance, max_loss_limit_usd, daily_loss_pct)
    except Exception as e:
        return f"Processing error: {e}", 500

    return render_template_string(RESULT_HTML, **results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
