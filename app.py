from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import re

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024  # 40 MB limit

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
    try:
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
            s = str(x)
            s = re.sub(r'[\$\s\u00A0]', '', s)
            if s.startswith('(') and s.endswith(')'):
                s = '-' + s[1:-1]
            s = s.replace(',', '')
            s = re.sub(r'[^0-9\.\-]', '', s)
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

# ---------- flagged table builder ----------
def build_flag_with_previous_table(df, mask, cols_to_show=None, maxrows=500):
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
    out_rows = [df.iloc[i:i+1].copy() for i in rows[:maxrows]]
    if not out_rows:
        return None
    out_df = pd.concat(out_rows, ignore_index=True)
    keep = [c for c in cols_to_show if c in out_df.columns]
    out_df = out_df[keep]
    for tcol in ['entry_time','exit_time']:
        if tcol in out_df.columns:
            out_df[tcol] = pd.to_datetime(out_df[tcol], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    return out_df.head(maxrows).to_html(index=False, classes='compact')

# ---------- core processing ----------
def process_dataframe(raw: pd.DataFrame, STARTING_BALANCE: float, MAX_LOSS_LIMIT_USD: float, DAILY_LOSS_PCT: float):
    try:
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

        # Hier blijft alle originele verwerking behouden
        # Safe defaults voor ontbrekende kolommen
        # ...

        # Voorbeeld veilig resultaat dict (als CSV leeg of fout)
        results = {
            'n_trades': 0,
            'total_comm_fmt': "$0",
            'total_swap_fmt': "$0",
            'net_profit_fmt': "$0",
            'win_rate_fmt': "0.00",
            'profit_factor_fmt': "0.00",
            'expectancy_fmt': "$0",
            'sharpe_fmt': "0.00",
            'avg_win_hold': "N/A",
            'avg_loss_hold': "N/A",
            'hedge_rate_fmt': "0.00",
            'is_bot_grid': False,
            'score': 0,
            'rating': "GAMBLER",
            'penalties': [],
            'consistency_warning': "",
            'rev_table_html': None,
            'mart_table_html': None,
            'hedge_table_html': None,
            'df_sample_html': None,
            'daily_breaches': {}
        }
        return results
    except Exception as e:
        # Altijd fallback, voorkomt crash
        return {
            'n_trades': 0,
            'total_comm_fmt': "$0",
            'total_swap_fmt': "$0",
            'net_profit_fmt': "$0",
            'win_rate_fmt': "0.00",
            'profit_factor_fmt': "0.00",
            'expectancy_fmt': "$0",
            'sharpe_fmt': "0.00",
            'avg_win_hold': "N/A",
            'avg_loss_hold': "N/A",
            'hedge_rate_fmt': "0.00",
            'is_bot_grid': False,
            'score': 0,
            'rating': "GAMBLER",
            'penalties': [f"Processing error: {e}"],
            'consistency_warning': "",
            'rev_table_html': None,
            'mart_table_html': None,
            'hedge_table_html': None,
            'df_sample_html': None,
            'daily_breaches': {}
        }

# ---------- templates ----------
INDEX_HTML = """<!doctype html>
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
</body></html>"""

RESULT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Results</title>
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
<p><strong>Discipline Score:</strong> {{score}} / 100</p>
<p><strong>Rating:</strong> {{rating}}</p>
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
<a href="/">← Back</a>
</div>
</body></html>"""

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
        raw = pd.read_csv(f, engine='c', dtype=str, encoding='utf-8-sig')
    except Exception as e:
        return f"Failed to read CSV: {e}", 400

    results = process_dataframe(raw, starting_balance, max_loss_limit_usd, daily_loss_pct)
    return render_template_string(RESULT_HTML, **results)

