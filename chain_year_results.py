"""
Chain monthly returns from 2025 CSVs to compute true year-end balance with compounding.
"""
import csv
from pathlib import Path

KEY_FIELDS = [
    'profile_be_stop','profile_be_trigger_r','profile_target_r','profile_max_trades_per_day',
    'weekly_days','monthly_days','min_trades','min_weekly_wr','min_monthly_wr',
    'min_weekly_nlr','min_monthly_nlr','top_n'
]

def main():
    files = [f'setup_results_2025-{m:02d}.csv' for m in range(1, 13)]
    monthly = []
    
    print('Loading CSVs...')
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f'  WARNING: {f} not found, skipping')
            continue
        rows = list(csv.DictReader(p.open()))
        month = rows[0].get('month', p.stem)
        keyed = {tuple(r[k] for k in KEY_FIELDS): r for r in rows}
        monthly.append((month, keyed))
        print(f'  Loaded {f}: {len(keyed)} setups')
    
    # Find common setups across all 12 months
    common = set(monthly[0][1].keys())
    for _, keyed in monthly[1:]:
        common &= set(keyed.keys())
    
    print(f'Total common setups: {len(common)}')
    
    # For each setup, chain the returns and compute year-end balance
    results = []
    for key in common:
        balance = 800.0
        profitable_months = 0
        worst_month = None
        month_detail = []
        
        for month, keyed in monthly:
            r = keyed[key]
            ret_pct = float(r['return_pct'])
            pnl = float(r['pnl_usd'])
            
            # Apply return as percentage compound
            balance = balance * (1.0 + ret_pct / 100.0)
            
            if pnl > 0:
                profitable_months += 1
            if worst_month is None or pnl < worst_month:
                worst_month = pnl
                
            month_detail.append((month, ret_pct, pnl, balance))
        
        year_end_pnl = balance - 800.0
        
        results.append({
            'key': key,
            'profitable_months': profitable_months,
            'year_end_balance': balance,
            'year_end_pnl': year_end_pnl,
            'worst_month': worst_month,
            'month_detail': month_detail
        })
    
    # Sort by year-end balance (best first)
    results.sort(key=lambda x: x['year_end_balance'], reverse=True)
    
    # Print top 12
    print('\n' + '='*80)
    print('TOP 12 SETUPS BY CHAINED YEAR-END BALANCE (2025)')
    print('='*80)
    
    for rank, r in enumerate(results[:12], 1):
        key = r['key']
        setup = {k: v for k, v in zip(KEY_FIELDS, key)}
        print(f'\nRANK {rank}')
        print(f'  Year-end balance: ${r["year_end_balance"]:.2f}')
        print(f'  Year-end PnL: ${r["year_end_pnl"]:+.2f}')
        print(f'  Profitable months: {r["profitable_months"]}/12')
        print(f'  Worst month PnL: ${r["worst_month"]:+.2f}')
        print(f'  Setup:')
        for field in KEY_FIELDS:
            print(f'    {field}: {setup[field]}')
        print('  Month-by-month:')
        for month, ret, pnl, bal in r['month_detail']:
            print(f'    {month}: return={ret:+.2f}%, pnl=${pnl:+.2f}, balance=${bal:.2f}')
    
    print('\n' + '='*80)

if __name__ == '__main__':
    main()
