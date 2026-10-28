# Author: Noe Blassel
# Last modified: Oct 28 2026

# runs equilibrium dynamics of the dimer system for t_eq LJ time units, aggregates samples for t_run, over n_traj independent trajectories
# Used for computing the free energy in (https://arxiv.org/abs/2610.36078, Figure 12)

# Usage (from repo root):
#   julia --project=dimer -t 8 dimer/eq_simulation.jl

using JLD2, Random
include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "system.jl"))

const n_traj = 8
const t_run = 500.0
const t_eq = 10.0
const dt = 1e-4
const log_every = 10

ξ_obs(sys, args...; kwargs...) = ξ(dimer_r(sys.coords, sys.boundary))

function main()
    nlog = round(Int, t_run / dt) ÷ log_every
    trajs = [Vector{Float32}(undef, nlog) for _ in 1:n_traj]
    seeds = rand(Xoshiro(20260910), UInt64, n_traj)

    Threads.@threads for ch in 1:n_traj
        rng = Xoshiro(seeds[ch])
        sys = dimer_system(; rng = rng)
        sim = overdamped(; dt = dt)
        simulate!(sys, sim, round(Int, t_eq / dt); run_loggers = false, n_threads = 1, rng = rng)
        sysl = System(sys; loggers = (xi = GeneralObservableLogger(ξ_obs, Float64, log_every),))
        simulate!(sysl, sim, round(Int, t_run / dt); run_loggers = true, n_threads = 1, rng = rng)
        v = values(sysl.loggers.xi)
        trajs[ch] = Float32.(@view v[1:min(nlog, length(v))])
    end
    out = joinpath(DATA, "dimer", "equilibrium_traj.jld2")
    mkpath(dirname(out))
    jldsave(out; dt = dt, log_stride = log_every, y_series = trajs)
    println("wrote $out")
end

main()
