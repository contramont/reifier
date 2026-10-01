"""and-of-parities (wave 3): parity of an integer s in [0, n] (a raw bit count or a count
feature) with floor((n-1)/2) units for every n >= 6, as c + sum_k max(0, gw s + gb)(vw s + vb),
with knots at irrational points between integers. Units are (gw, gb, vw, vb) plus a constant c.

FORMS       variable-projection forms (search/best1d.py), polished to float64, smallest
            cancelling terms found: n = 6, 8, 10, 14, 16
FORMS_FLAT  the flattest forms found at the integers (count inputs): n = 10, 11
ext(n)      even n: the n = 6 form (knots 3 -+ (2 sqrt 2 - 2)) + glu_xor ramps to its right
ext_c(n)    the same with the n = 6 core in the middle and ramps on both sides
ext_t(n)    the same with the core at the top (its non-flat integers are n - 4, n - 2)
Gates are scaled so that every integer is >= 1 away from each knot in gate units. Exact in
float64 to ~1e-12 at every integer (check(); 50-digit check in search/mpcheck_forms.py)."""

FORMS = {
    6: ([(5.828427124746197, -12.656854249492394, -0.17157287525380965, 1.3431457505076183), (-5.828427124746182, 22.313708498984727, 0.1715728752538105, 0.3137084989847608)], -7.000000000000003),  # K=2 knots [2.1716, 3.8284] max term 8.0
    8: ([(3.5793753029253734, -24.055627120477613, -2.2530368290867, 19.771257803606684), (-5.828427124746091, 22.313708498984365, 0.17157287525379708, 0.3137084989847353), (5.828427124746303, -12.656854249492605, -0.17157287525379414, 1.3431457505074818)], -6.999999999999256),  # K=3 knots [6.7206, 3.8284, 2.1716] max term 8.0
    10: ([(-4.768069363494745, 8.53613872698949, 0.08133016036572861, 1.1129153712044226), (12.16441400296914, -47.65765601187656, -0.08220700148448662, 0.8288280059379367), (11.783365227411691, -95.26692181929353, -0.01643754803487898, 0.5188796481552542), (-2.929311422413384, 16.575868534480303, 0.4267214439665662, -0.060328663799418376)], -8.49999999999994),  # K=4 knots [1.7903, 3.9178, 8.0849, 5.6586] max term 9.5 max slope 3.25
    14: ([(2.6339279428723796, -11.535711771489519, -0.49698689357371467, 5.448885784811727), (6.164508343229408, -13.329016686458816, -0.06836578289267206, 0.9796144438520383), (5.879834439704833, -46.038675517638666, 0.12423293763608285, 0.11639098916939813), (5.42788705641034, -66.13464467692408, -0.0916584936171165, 2.094925825634193), (-6.9810972771186455, 42.88658366271187, 0.08511398867906542, 0.6516217189604046), (-6.349658660807075, 62.49658660807075, 0.06391067411115532, 0.17251518966448443)], -38.727439858709936),  # K=6 knots [4.3797, 2.1622, 7.8299, 12.1842, 6.1432, 9.8425] max term 67.3
    16: ([(-15.985230726927206, 190.82276872312647, 0.054926602157799945, -0.2782520863086247), (-2.44243807862077, 5.88487615724154, -0.7138149405944808, 4.9145274651819095), (5.939195203487422, -82.14873284882391, 0.07444562152844554, -0.1579445447935271), (-3.7073839112243996, 38.073839112243995, -0.11112511963995174, 3.370707030496394), (3.5618343388525986, -29.49467471082079, -0.25497480223839963, 3.8561544302169892), (-6.657210792276781, 38.943264753660685, 0.34209790484418007, -1.0034273080735137), (3.24096908718164, -13.96387634872656, -0.1647559905027815, 4.164699569307431)], -65.08357384582918),  # K=7 knots [11.9374, 2.4094, 13.8316, 10.2697, 8.2808, 5.8498, 4.3085] max term 128.3
}


# the flattest forms found (n = 10: max slope 2.216 at the integers, max term 10.6, nearest
# knot 0.016 from an integer; n = 11 below): for count inputs that carry noise
FORMS_FLAT = {11: ([(-5.221382662939292, 16.664147988817877, 0.6210791515887776, -2.470158397646014), (5.125791341375455, -11.25158268275091, -0.11125428269581947, 1.303273826475105), (-7.202146402116476, 35.01073201058238, -0.09339239136930357, -0.27136845124470327), (15.463577322631808, -109.24504125842266, 0.10154611744559863, -1.0889256938707226), (-2.8144678561694305, 24.330210705524873, -0.5579265137070489, 2.258909017248795)], -4.295839137423782),  # n = 11: max slope 2.000 (MIN_PARITY[11]: 2.846), max term 55.0
              10: ([(63.4031912338525, -508.22552987082, -0.0005095074248798201, 0.06868485278848081), (-10.727649349599163, 22.455298699198327, -0.020102323531834845, 0.47150600298710665), (10.381017057854676, -42.5240682314187, -0.09632967506236358, 0.9540173443260149), (-3.6198963001369844, 20.719377800821906, 0.3358247245689639, -0.15234563556691283)], -7.431301355923391)}



def ext(n):
    """even n >= 6: the 2-unit n = 6 form (its last piece is 1 - (s - 5)^2 through s = 4, 5, 6)
    continued by glu_xor ramps 4 max(0, s - 6 - 2j) (knots on even integers: flat under silu,
    dyadic weights): n/2 - 1 units, one fewer than glu_xor, for every even n"""
    assert n % 2 == 0 and n >= 6
    us, c = FORMS[6]
    return list(us) + [(1.0, -(6.0 + 2 * j), 0.0, 4.0) for j in range((n - 6) // 2)], c


FORMS_EXT = {n: ext(n) for n in range(8, 66, 2)}


def ext_c(n):
    """even n >= 6: the n = 6 form moved to the middle of [0, n] (it covers s0 .. s0 + 6, s0 even
    and near n/2 - 3) with glu_xor ramps on both sides: 4 max(0, s - s0 - 6 - 2j) to the right
    and 4 max(0, s0 - 2j - s) to the left (the form's first piece is 1 - (s - s0 - 1)^2).
    n/2 - 1 units as ext(n), cancelling terms ~ (n/2)^2 instead of ~ n^2 (as dl3's centred)."""
    assert n % 2 == 0 and n >= 6
    s0 = 2 * ((n - 6) // 4)
    us, c = FORMS[6]
    out = [(gw, gb - gw * s0, vw, vb - vw * s0) for gw, gb, vw, vb in us]
    out += [(1.0, -(s0 + 6.0 + 2 * j), 0.0, 4.0) for j in range((n - s0 - 6) // 2)]
    out += [(-1.0, float(s0 - 2 * j), 0.0, 4.0) for j in range(s0 // 2)]
    return out, c


FORMS_EXTC = {n: ext_c(n) for n in range(8, 66, 2)}


def ext_t(n):
    """even n >= 6: the n = 6 form at the top of the range (s0 = n - 6), glu_xor ramps to its left:
    the points with a nonzero slope (s0, s0 + 2, s0 + 4) sit where counts are rare"""
    assert n % 2 == 0 and n >= 6
    s0 = n - 6
    us, c = FORMS[6]
    out = [(gw, gb - gw * s0, vw, vb - vw * s0) for gw, gb, vw, vb in us]
    out += [(-1.0, float(s0 - 2 * j), 0.0, 4.0) for j in range(s0 // 2)]
    return out, c


FORMS_EXTT = {n: ext_t(n) for n in range(8, 66, 2)}


def check(n, forms=None):
    us, c = (forms or FORMS)[n]
    return max(abs(c + sum(max(0.0, gw * s + gb) * (vw * s + vb) for gw, gb, vw, vb in us) - s % 2) for s in range(n + 1))


if __name__ == "__main__":
    for n in FORMS:
        print(n, len(FORMS[n][0]), check(n))
    for n in FORMS_FLAT:
        print("flat", n, len(FORMS_FLAT[n][0]), check(n, FORMS_FLAT))
    for n in FORMS_EXT:
        print("ext", n, len(FORMS_EXT[n][0]), check(n, FORMS_EXT))
    for n in FORMS_EXTT:
        print("extt", n, len(FORMS_EXTT[n][0]), check(n, FORMS_EXTT))
    for n in FORMS_EXTC:
        us, c = FORMS_EXTC[n]
        big = max(abs(max(0.0, gw * s + gb) * (vw * s + vb)) for gw, gb, vw, vb in us for s in range(n + 1))
        print("extc", n, len(us), check(n, FORMS_EXTC), "max term", round(big, 1))
