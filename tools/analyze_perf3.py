# -*- coding: utf-8 -*-
"""三版本收敛对比分析：gso_100_{exact,grid,gpu}.out"""
import sys

def read_gso(p):
    luc, sc, tr = [], [], []
    for ln in open(p):
        if ln.startswith('#'): continue
        parts = ln.split()
        if len(parts) < 10: continue
        try:
            # find token ending with ')' -> the last coord token
            close = next(i for i, t in enumerate(parts) if t.endswith(')'))
            # parts[close+1]=RecID parts[close+2]=LigID parts[close+3]=luciferin
            luc.append(float(parts[close + 3]))
            sc.append(float(parts[-1]))
            x = float(parts[0].strip('(').rstrip(','))
            y = float(parts[1].rstrip(','))
            z = float(parts[2].rstrip(','))
            tr.append((x, y, z))
        except Exception:
            pass
    return luc, sc, tr

def stats(label, p):
    luc, sc, tr = read_gso(p)
    if not luc:
        print(f'{label}: 解析失败 ({p})'); return None
    sl = sorted(luc); ss = sorted(sc)
    best_i = sc.index(min(sc))
    # top-5 by scoring (most negative first = best in LKlight rank convention)
    order = sorted(range(len(sc)), key=lambda i: sc[i])
    top5 = [tr[i] for i in order[:5]]
    print(f'[{label}] {len(luc)} 解')
    print(f'  luciferin 最小 {min(luc):.2f} / 中位 {sl[len(sl)//2]:.2f}')
    print(f'  scoring   最小 {min(sc):.2f} / 中位 {ss[len(ss)//2]:.2f}')
    print(f'  best-scoring pose: ({tr[best_i][0]:.2f}, {tr[best_i][1]:.2f}, {tr[best_i][2]:.2f})')
    return dict(luc=min(luc), sc=min(sc), med=sorted(sc)[len(sc)//2], top5=top5)

if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else '/tmp'
    res = {}
    for v in ['exact', 'grid', 'gpu']:
        r = stats(v, f'{base}/gso_100_{v}.out')
        if r: res[v] = r
    if len(res) == 3:
        print('\n=== 相对 exact 偏差 ===')
        for v in ['grid', 'gpu']:
            print(f'{v}: best-scoring {res[v]["sc"] - res["exact"]["sc"]:+.2f} '
                  f'(exact {res["exact"]["sc"]:.2f}), '
                  f'luciferin {res[v]["luc"] - res["exact"]["luc"]:+.2f}')
        # top-5 平移解重合度（容差 2 Å）
        def close(a, b, tol=2.0):
            return all(abs(a[i] - b[i]) <= tol for i in range(3))
        for v in ['grid', 'gpu']:
            hit = sum(1 for a in res[v]['top5'] if any(close(a, b) for b in res['exact']['top5']))
            print(f'{v} top5 与 exact top5 重合(±2Å): {hit}/5')
