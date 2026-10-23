# Author: Noe Blassel
# Last modified: Oct 23 2026

# defines the triple-well potential W in the dimer length r₁₂

const R_MID = 1.45
const D_SPC = 0.7
const K_W = 78.0
const C_W = 0.6
const LAM_W = 0.49
const S_CAP = 1.1                                          # W is linearly extended beyond |s| = S_CAP

@inline s_of_r(r) = (r - R_MID) / D_SPC
@inline r_of_s(s) = R_MID + D_SPC * s

@inline _w_poly(s) = K_W * (s^6 / 6 - (1 + C_W^2) * s^4 / 4 + C_W^2 * s^2 / 2) + LAM_W * s
@inline _dw_poly(s) = K_W * s * (s^2 - 1) * (s^2 - C_W^2) + LAM_W

@inline function w_of_s(s)
    abs(s) <= S_CAP && return _w_poly(s)
    sc = sign(s) * S_CAP
    return _w_poly(sc) + _dw_poly(sc) * (s - sc)
end

@inline function dw_ds(s)
    abs(s) <= S_CAP && return _dw_poly(s)
    return _dw_poly(sign(s) * S_CAP)
end

@inline W_of_r(r) = w_of_s(s_of_r(r))
@inline dW_dr(r) = dw_ds(s_of_r(r)) / D_SPC


# finds saddle points of W by dichotomy root-searching on the derivative
function saddles(; ngrid::Int = 20_001)
    ss = range(-0.999, 0.999; length = ngrid)
    roots = Float64[]
    for i in 1:(ngrid - 1)
        f1, f2 = dw_ds(ss[i]), dw_ds(ss[i + 1])
        (f1 > 0 && f2 < 0) || continue
        a, b = ss[i], ss[i + 1]
        for _ in 1:200
            m = (a + b) / 2
            dw_ds(m) > 0 ? (a = m) : (b = m)
        end
        push!(roots, (a + b) / 2)
    end
    length(roots) == 2 || error("expecting two saddle points, found $(length(roots)): $roots")
    return r_of_s(roots[1]), r_of_s(roots[2])
end

const R_LO, R_HI = saddles()
const OMEGA_W = R_HI - R_LO

@inline y_of_r(r) = (r - R_LO) / OMEGA_W
@inline in_omega(r) = R_LO < r < R_HI