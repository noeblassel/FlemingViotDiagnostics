# Author: Noe Blassel
# Last modified: Oct 28 2026

# defines the triple-well interaction W in the dimer length r₁₂, forces and computes landscape parameters

## Constants in the triple-well potential
const r_middle = 1.45
const s_denom = 0.7
const K_W = 78.0
const C_W = 0.6
const E_W = 0.49
const s_cap = 1.1                                          # W is linearly extended for |s| > s_cap

@inline s_of_r(r) = (r - r_middle) / s_denom
@inline r_of_s(s) = r_middle + s_denom * s

@inline w_poly(s) = K_W * (s^6 / 6 - (1 + C_W^2) * s^4 / 4 + C_W^2 * s^2 / 2) + E_W * s
@inline dw_poly(s) = K_W * s * (s^2 - 1) * (s^2 - C_W^2) + E_W

@inline function w_of_s(s)
    abs(s) <= s_cap && return w_poly(s)
    sc = sign(s) * s_cap
    return w_poly(sc) + dw_poly(sc) * (s - sc)
end

@inline function dw_ds(s)
    abs(s) <= s_cap && return dw_poly(s)
    return dw_poly(sign(s) * s_cap)
end

@inline w_of_r(r) = w_of_s(s_of_r(r))
@inline dw_dr(r) = dw_ds(s_of_r(r)) / s_denom

width_Ω

# finds saddle points of W by root-searching dichotomy on the derivative dW
function w_saddles(; ngrid::Int = 20_001)
    srange = range(-0.999, 0.999; length = ngrid)
    roots = Float64[]
    for i in 1:(ngrid - 1)
        f1, f2 = dw_ds(srange[i]), dw_ds(srange[i + 1])
        (f1 > 0 && f2 < 0) || continue
        a, b = srange[i], srange[i + 1]
        for _ in 1:200
            m = (a + b) / 2
            dw_ds(m) > 0 ? (a = m) : (b = m)
        end
        push!(roots, (a + b) / 2)
    end
    length(roots) == 2 || error("expecting two saddle points, found only $(length(roots)): $roots")
    return r_of_s(roots[1]), r_of_s(roots[2])
end

const r_L, r_R = w_saddles()
const width_Ω = r_R - r_L

@inline ξ(r) = (r - r_L) / width_Ω
@inline in_metastable_state(r) = r_L < r < r_R