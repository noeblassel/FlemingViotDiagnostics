# Author: Noe Blassel
# Last modified: Oct 23 2026

# Equilibriates the dimer system for T_EQ LJ time units, then aggregates samples for T_RUN, over NCHAIN independent trajectories
# of the equilibrium dynamics (with D_d = 1). Used for computing the free energy in Figure 12

#   julia --project=dimer -t 8 dimer/characterise.jl

using JLD2, Random
include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "system.jl"))

const NCHAIN = 8
const T_RUN = 500.0
const T_EQ = 10.0
const DT = 1e-4
const LOG_STRIDE = 10

y_obs(sys, args...; kwargs...) = y_of_r(dimer_r(sys.coords, sys.boundary))

function main()
    nlog = round(Int, T_RUN / DT) ÷ LOG_STRIDE
    trajs = [Vector{Float32}(undef, nlog) for _ in 1:NCHAIN]
    seeds = rand(Xoshiro(20260910), UInt64, NCHAIN)

    Threads.@threads for ch in 1:NCHAIN # parallelize over trajectories, not atoms
        rng = Xoshiro(seeds[ch])
        sys = build_system(; rng = rng)
        sim = overdamped(; dt = DT)
        simulate!(sys, sim, round(Int, T_EQ / DT); run_loggers = false, n_threads = 1, rng = rng)
        sysl = System(sys; loggers = (y = GeneralObservableLogger(y_obs, Float64, LOG_STRIDE),))
        simulate!(sysl, sim, round(Int, T_RUN / DT); run_loggers = true, n_threads = 1, rng = rng)
        v = values(sysl.loggers.y)
        trajs[ch] = Float32.(@view v[1:min(nlog, length(v))])
    end
    out = joinpath(DATA, "dimer", "equilibrium_traj.jld2")
    mkpath(dirname(out))
    jldsave(out; dt = DT, log_stride = LOG_STRIDE, y_series = trajs)
    println("wrote $out")
end

main()
