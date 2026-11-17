# Author: Noe Blassel
# Last modified: Nov 17 2026

# 

using Random, LinearAlgebra
isdefined(@__MODULE__, :dimer_system) || include(joinpath(@__DIR__, "system.jl"))

site_index(i, j, k) = 1 + i + n_at_perdim* j + n_at_perdim^2 * k

"""
    place_dimer(r)

Lattice configuration with the dimer atoms on the two sites nearest the box centre, moved
apart along x to separation `r`.
"""
function place_dimer(r::Real)
    coords = lattice_coords()
    i = (n_at_perdim-1) ÷ 2
    s1, s2 = site_index(i, i, i), site_index(i+1, i, i)
    coords[1], coords[s1] = coords[s1], coords[1]
    coords[2], coords[s2] = coords[s2], coords[2]
    m = (coords[1] + coords[2]) / 2
    coords[1] = m - SVector(r / 2, 0.0, 0.0)
    coords[2] = m + SVector(r / 2, 0.0, 0.0)
    return coords
end

"""
    relax_solvent!(sys; frozen = (1, 2), n_descent = 2000, cap = 0.01, n_thermal = 200_000,
                   dt = 1e-4, rng = Random.default_rng(), n_threads = 1)

Initial condition preparation: with atomic coordinates with `frozen` indices fixed, do:
- `n_descent` steps of steepest descent, aborted if the maximum per-atom displacement becomes smaller than `cap`
- `n_thermal` steps of Euler-Maruyama with `dt` for the overdamped Langevin dynamics with kT = 1
"""
function relax_solvent!(sys; frozen = (1, 2), n_descent::Int = 2000, cap::Float64 = 0.01,
                        n_thermal::Int = 200_000, dt::Float64 = 1e-4,
                        rng::AbstractRNG = Random.default_rng(), n_threads::Int = 1)
    c = sys.coords
    f = Molly.zero_forces(sys)
    bufs = Molly.init_buffers!(sys, n_threads)
    nbrs = find_neighbors(sys, sys.neighbor_finder; n_threads = n_threads)

    for k in 1:n_descent # steepest descent
        Molly.forces!(f, sys, nbrs, bufs, Val(false), k; n_threads = n_threads)
        fmax = maximum(norm, f)
        scale = fmax > 0 ? min(1e-4, cap / fmax) : 1e-4
        @inbounds for a in eachindex(c)
            a in frozen && continue
            c[a] = wrap_coords(c[a] + f[a] * scale, sys.boundary)
        end
        nbrs = find_neighbors(sys, sys.neighbor_finder, nbrs, k, true; n_threads = n_threads)
    end

    σ = sqrt(2 * dt * 1.0 / 1.0)
    for k in 1:n_thermal # thermalization
        Molly.forces!(f, sys, nbrs, bufs, Val(false), k; n_threads = n_threads)
        @inbounds for a in eachindex(c)
            a in frozen && continue
            c[a] = wrap_coords(c[a] + f[a] * dt + σ * SVector(randn(rng), randn(rng), randn(rng)), sys.boundary)
        end
        nbrs = find_neighbors(sys, sys.neighbor_finder, nbrs, k; n_threads = n_threads)
    end
    return sys
end

"""
    solvated(r; seed, kwargs...)

Configuration with the dimer at separation `r` and the solvent relaxed by
[`relax_solvent!`](@ref) from `Xoshiro(seed)`.
"""
function solvated(r::Real; seed::Integer, kwargs...)
    rng = Xoshiro(seed)
    sys = dimer_system(; coords = place_dimer(r), rng = rng)
    relax_solvent!(sys; rng = rng, kwargs...)
    abs(dimer_r(sys.coords, sys.boundary) - r) < 1e-10 || error("dimer moved during solvation")
    return copy(sys.coords)
end
