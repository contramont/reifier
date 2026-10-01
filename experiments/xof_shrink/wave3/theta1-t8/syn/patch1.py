src = open('pbcheck.py').read()
old = """        tv = sp.nsimplify(tv)
        ok = (T0 - 1 < tv <= T0) if dr > 0 else (T0 <= tv < T0 + 1)"""
new = """        if not tv.is_real:
            continue
        tv = sp.nsimplify(tv)
        ok = bool((T0 - 1 < tv <= T0) if dr > 0 else (T0 <= tv < T0 + 1))"""
assert old in src
open('pbcheck.py','w').write(src.replace(old, new))
