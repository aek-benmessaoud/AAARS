import json
from scipy.stats import binom

recs = json.load(open(r'F:\Project11-AAARS\results\raw\policies_results.json'))
THR = 95.0


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    pv = 2 * binom.cdf(k, n, 0.5)
    return min(1.0, pv)


for pol in ['frontier', 'greedy', 'hotspot']:
    sub = [r for r in recs if r.get('alloc') == pol]
    a = b = c = d = 0
    for r in sub:
        fa = r['aaars__recall'] < THR
        fc = r['chao1_ci__recall'] < THR
        if not fc and not fa:
            a += 1
        elif fc and fa:
            d += 1
        elif fc and not fa:
            b += 1
        elif not fc and fa:
            c += 1
    n = b + c
    pv = mcnemar_p(b, c)
    mr_a = sum(r['aaars__recall'] for r in sub) / len(sub)
    mr_c = sum(r['chao1_ci__recall'] for r in sub) / len(sub)
    fa_cnt = sum(1 for r in sub if r['aaars__recall'] < THR)
    fc_cnt = sum(1 for r in sub if r['chao1_ci__recall'] < THR)
    print(f'{pol:9s} McNemar b={b} c={c} n={b+c} p={pv:.4f} | FC AA={fa_cnt} CH={fc_cnt}')
    print(f'          meanRecall(all80) : Chao1={mr_c:.1f}  AAARS={mr_a:.1f}')
