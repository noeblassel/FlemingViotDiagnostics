# Author: Noe Blassel
# Last modified: Nov 17 2026

# Generates initial configurations for the dimer experiments, 

# usage (from repo root): julia --project=dimer -t 1 dimer/make_initial_conditions.jl

using JLD2, Random
include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "initial_conditions.jl"))

const STATES = [("compact", 0.740), ("middle", 1.435), ("extended", 2.146)]   # minima of the free energy

function main()
    mkpath(joinpath(DATA, "dimer"))
    coords = [solvated(r; seed = seed_ic() + i) for (i, r) in enumerate(initial_dimer_lengths)]
    for (i, c) in enumerate(coords)
        println("point $(initial_config_labels[i]): r = ", dimer_r(c, CubicBoundary(box_length, box_length, box_length)))
    end
    jldsave(joinpath(DATA, "dimer", "initial_conditions.jld2"); init_r = initial_dimer_lengths, init_label = initial_config_labels,
            coords = coords)

    snapshots = [reduce(hcat, [Vector(c) for c in solvated(r; seed = seed_ic() + 100 + k)])
             for (k, (_, r)) in enumerate(STATES)]
    jldsave(joinpath(DATA, "dimer", "snapshot_states.jld2"); names = first.(STATES), r = last.(STATES),
            coords = snapshots, boxl = box_length)
    println("wrote ", joinpath(DATA, "dimer", "{initial_conditions,snapshot_states}.jld2"))
end

main()
