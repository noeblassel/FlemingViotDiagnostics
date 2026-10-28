# Author: Noe Blassel
# Last modified: Oct 28 2026

# shared parameters for the dimer experiments of (https://arxiv.org/abs/2610.36078, Section 4.4.)

const sim_dt = 1e-4

## experimental settings
"""
Experimental settings:
`d_dimer`: mobility of the dimer particles (D_d)
`stride_ref`: time-resolution for the definition of reference decorrelation times `t_0(f)` (in number of timesteps)
`hold`: holding time (t_h) used in the definition of reference decorrelation times `t_0(f)`
`nc_range`: clamp-normalization range for the soft coordination number
"""
const settings = Dict("slow" => (d_dimer = 0.05, stride_ref = 40, hold = 0.2, nc_range = (4.0, 10.0)),
                  "fast" => (d_dimer = 1.0, stride_ref = 25, hold = 0.05, nc_range = (4.0, 10.0)))

function arm(name::AbstractString)
    haskey(settings, name) || throw(ArgumentError("unknown setting \"$name\", expected one of $(collect(keys(settings)))"))
    return settings[name]
end

## initial configurations

const initial_dimer_lengths = [1.10, 1.25, 1.45, 1.65, 1.82, 1.82]
const initial_config_labels = ["1", "2", "3", "4", "5", "5′"]

const τ_grid = [100, 200, 400] # lag times                        
const n_grid = [25, 50, 100] # number of replicas
const n_fv_trajectories = 20                                           # Fleming-Viot simulations per grid cell
const n_corr_dimer = 10                                      

const N_ref = 72 # number of Fleming-Viot replicas for reference sampling
const Tsim_ref = 3.0 # time horizon for reference sampling
const nsamps_ref = 3000 # reference runs per point

const nbins_max = 256 # high-resolution histogram bin-count (which will coarsened when computing features)

## reproducible seeds for RNG

seed_ic() = 20260910
seed_ref(setting, point, run) = hash((:reference, setting, point, run))
seed_grid(setting, cell, traj) = hash((:grid, setting, cell, traj))
seed_exit(setting, ids...) = hash((:exit, setting, ids...))

function cell_to_idx(cell::Int)
    c = cell - 1
    return (mod(c, 3) + 1, mod(div(c, 3), 3) + 1, div(c, 9) + 1)
end
n_cells() = length(τ_grid) * length(n_grid) * length(initial_dimer_lengths)


## datafile utilities
const DATA = normpath(joinpath(@__DIR__, "..", "data"))
dimer_setting_dir(name) = joinpath(DATA, "dimer", name)

## parsing keyword arguments
function getarg(args, key, default)
    i = findfirst(a -> startswith(a, "--$key="), args)
    i === nothing && return default
    v = split(args[i], "="; limit = 2)[2]
    return default isa AbstractString ? String(v) : parse(typeof(default), v)
end
