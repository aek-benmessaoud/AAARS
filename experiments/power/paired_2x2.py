import json
import numpy as np

RAW = r'F:\Project11-AAARS\results\raw\power_revision.json'
d = json.load(open(RAW))
mr = [r for r in d if r.get('alloc') == 'minerich']
N = len(mr)
THR = 95.0


def is_fc(r, m):
    return (r.get(f'{m}__t') is not None and r[f'{m}__recall'] < THR)


# 2x2: rows Chao1, cols AAARS
a = b = c = dd = 0
for r in mr:
    fc = is_fc(r, 'chao1_ci')
    fa = is_fc(r, 'aaars')
    if not fc and not fa:
        a += 1
    elif fc and not fa:
        b += 1
    elif not fc and fa:
        c += 1
    else:
        dd += 1

print(f'N={N}  2x2 (Chao1 FC in {b+dd}, AAARS FC in {c+dd})')
print('        AAARS safe   AAARS FC')
print(f'Chao1 safe   {a:5d}        {c:5d}')
print(f'Chao1 FC     {b:5d}        {dd:5d}')
print('discordant b (AAARS corrects) =', b, ' c (AAARS makes worse) =', c)
print('FC%: Chao1', 100*(b+dd)/N, ' AAARS', 100*(c+dd)/N, ' diff (Chao-AAARS) =',
      100*((b+dd) - (c+dd))/N, 'pp')

# Paired bootstrap CI for the difference in FC counts (episode-level resample)
rng = np.random.default_rng(0)
pairs = []
for r in mr:
    pairs.append((is_fc(r, 'chao1_ci'), is_fc(r, 'aaars')))
M = 20000
diffs = np.empty(M)
for i in range(M):
    idx = rng.integers(0, N, size=N)
    ch = sum(1 for j in idx if pairs[j][0])
    aa = sum(1 for j in idx if pairs[j][1])
    diffs[i] = (ch - aa)  # per-episode count difference
# diff in FC RATE = mean per episode
rate_diffs = diffs / N * 100.0
lo, hi = np.percentile(rate_diffs, [2.5, 97.5])
print(f'paired bootstrap 95% CI [diff rate (Chao-AAARS), pp]: [{lo:.1f}, {hi:.1f}]')
print(f'mean diff rate: {rate_diffs.mean():.2f} pp')
