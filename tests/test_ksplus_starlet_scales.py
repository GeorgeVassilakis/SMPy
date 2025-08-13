from smpy.filters.starlet import (
    compute_starlet_nscales_max,
    starlet_nscales_support_aware,
)


def coarsest_support(nscales: int) -> int:
    # nscales = J + 1 => J = nscales - 1, coarsest detail j = J - 1 = nscales - 2
    j = nscales - 2
    return 4 * (2 ** j) + 1


def test_nscales_max_examples():
    # Spot-check known sizes and ensure coarsest support fits in the image.
    # For power-of-two sizes, make sure we do not overshoot.
    nmax_128 = compute_starlet_nscales_max(128, 128)
    assert nmax_128 == 6
    assert coarsest_support(nmax_128) <= 128

    nmax_256 = compute_starlet_nscales_max(256, 256)
    assert nmax_256 == 7
    assert coarsest_support(nmax_256) <= 256

    nmax_300 = compute_starlet_nscales_max(300, 300)
    assert coarsest_support(nmax_300) <= 300


def test_support_aware_auto_vs_override():
    for N in [64, 128, 256, 300]:
        auto_n = starlet_nscales_support_aware(N, N, cfg_nscales=None)
        assert coarsest_support(auto_n) <= N

        # Oversized override should be clipped
        clipped_n = starlet_nscales_support_aware(N, N, cfg_nscales=999)
        assert clipped_n == auto_n

        # Below minimum should be raised to at least 2
        min_n = starlet_nscales_support_aware(N, N, cfg_nscales=1)
        assert min_n >= 2
        assert coarsest_support(min_n) <= N

