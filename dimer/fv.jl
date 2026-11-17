# Author: Noe Blassel
# Last modified: Nov 17 2026

# solvated dimer simulations of the fleming-viot particle system

using Random, LinearAlgebra, Statistics
isdefined(@__MODULE__, :step!) || include(joinpath(@__DIR__, "system.jl"))

# branching replicas
function copy_neighbors!(dst, src)
    length(dst.list) == length(src.list) || resize!(dst.list, length(src.list))
    copyto!(dst.list, src.list)
    dst.n = src.n
    return dst
end

"""
    run_fv(q0; N, stride, nframes, d_dimer, dt = 1e-4, seed = 1, nbins = 256,
           snapshot_frames = Int[], record_hist = true, nc_range = (0.0, 1.0))

Simulates one Fleming–Viot trajectory of `N` replicas started at `q0`, collecting samples into `nframes` frames of `stride` steps.
Full-atom coordinates are recorded for frame indices contained in `snapshot_frames`

Returns a NamedTuple with
  - `xi`: `N × nframes`, the value of ξ of each replica at each frame
  - `hist`: `nbins × nframes`, histogram of the `N × stride` values of ξ in each frame's window
  - `ncoord`: `N × nframes`, soft coordination number of each replica at each frame
  - `gr`: Gelman–Rubin statistic minus one of `y` at each frame
  - `snapshots`: replica coordinates recorded at each `snapshot_frames`
  - `n_branch`: number of branching events

If `record_hist_nc=true`, histograms of the soft coordination number's are returned in `hist_nc` (clamped
to `nc_range`) and the history of its Gelman–Rubin statistic in `gr_nc`.
"""
function run_fv(q0; N::Int, stride::Int, nframes::Int, d_dimer::Float64, dt::Float64 = 1e-4,
                seed::Integer = 1, nbins::Int = 256, snapshot_frames = Int[], record_hist_nc::Bool = true,
                nc_range = (0.0, 1.0))
    master = Xoshiro(seed) # master rng
    syss = [dimer_system(; coords = q0, rng = Xoshiro(rand(master, UInt64))) for _ in 1:N]
    rngs = [Xoshiro(rand(master, UInt64)) for _ in 1:N] # derived rngs
    its = [Integrator(syss[i]; dt = dt, d_dimer = d_dimer) for i in 1:N]

    ξs = [ξ(dimer_r(syss[i].coords, syss[i].boundary)) for i in 1:N]
    all(0 .< ξs .< 1) || error("initial configuration outside Ω (ξ = $(ξs[1]))")

    xi = Matrix{Float32}(undef, N, nframes) # Float32 for RNN inference
    hist_ξ = zeros(Int32, record_hist_nc ? nbins : 0, nframes) # ξ histogram
    hist_nc = zeros(Int32, record_hist_nc ? nbins : 0, nframes) # soft coordinate number histogram
    ncrd = Matrix{Float32}(undef, N, nframes) # coordination number frames Float32 for RNN inference
    gr_hist_ξ = Vector{Float64}(undef, nframes)
    gr_hist_nc = Vector{Float64}(undef, record_hist_nc ? nframes : 0)
    snaps = Dict{Int, Vector{typeof(q0)}}()

    sum_lin_ξ = zeros(N)
    sum_sq_ξ = zeros(N)
    nc = zeros(N)
    sum_lin_nc = zeros(N)
    sum_sq_nc = zeros(N)

    nkilled = 0
    
    for k in 1:(stride * nframes)
        for i in 1:N
            ξs[i] = step!(syss[i], its[i], rngs[i], k)
        end

        survivors = findall(v -> 0 < v < 1, ξs)
        isempty(survivors) && error("Extinction event at step $k")
        if length(survivors) < N
            for i in 1:N
                (0 < ξs[i] < 1) && continue
                j = survivors[rand(master, 1:length(survivors))]
                copyto!(syss[i].coords, syss[j].coords)
                copy_neighbors!(its[i].nbrs[], its[j].nbrs[])
                ξs[i] = ξs[j]
                nkilled += 1
            end
        end

        @inbounds for i in 1:N
            sum_lin_ξ[i] += ξs[i]
            sum_sq_ξ[i] += ξs[i]^2
            hist_ξ[bin_of(ξs[i], nbins), cld(k, stride)] += one(Int32)
            record_hist_nc || continue
            sum_lin_nc[i] += nc[i]
            sum_sq_nc[i] += nc[i]^2
            nc[i] = soft_ncoord(syss[i].coords, syss[i].boundary)
            hist_nc[bin_of((nc[i] - nc_range[1]) / (nc_range[2] - nc_range[1]), nbins), cld(k, stride)] += one(Int32)
        end

        if mod1(k, stride) == stride
            frame_ix = k ÷ stride
            gr_hist_ξ[frame_ix] = (sum(sum_sq_ξ) - sum(sum_lin_ξ)^2 / (N * k)) / sum(sum_sq_ξ .- (sum_lin_ξ .^ 2) ./ k) - 1
            record_hist_nc &&
                (gr_hist_nc[frame_ix] = (sum(sum_sq_nc) - sum(sum_lin_nc)^2 / (N * k)) / sum(sum_sq_nc .- (sum_lin_nc .^ 2) ./ k) - 1)
            @inbounds for i in 1:N
                xi[i, frame_ix] = Float32(ξs[i])
                ncrd[i, frame_ix] = Float32(record_hist_nc ? nc[i] : soft_ncoord(syss[i].coords, syss[i].boundary))
            end
            frame_ix in snapshot_frames && (snaps[frame_ix] = [copy(s.coords) for s in syss])
        end
    end

    return (xi = xi, hist = hist_ξ, ncoord = ncrd, gr = gr_hist_ξ, hist_nc = hist_nc, gr_nc = gr_hist_nc,
            snapshots = snaps, n_branch = nkilled)
end

"""
    run_to_exit(q0; d_dimer, dt, seed, max_steps) -> (t_exit, side)

    Runs sequential dynamics of the dimer system with D_d = `d_dimer` from initial configuration `q0`, until exit from the state or `max_steps` integration steps are performed
"""
function run_to_exit(q0; d_dimer::Float64, dt::Float64, seed::Integer, max_steps::Int)
    rng = Xoshiro(seed)
    sys = dimer_system(; coords = copy(q0), rng = Xoshiro(rand(rng, UInt64)))
    it = Integrator(sys; dt = dt, d_dimer = d_dimer)
    xi = ξ(dimer_r(sys.coords, sys.boundary))
    0 < y < 1 || error("start is outside Ω (ξ = $xi)")
    for k in 1:max_steps
        xi = step!(sys, it, rng, k)
        xi <= 0 && return (k * dt, -1)
        xi>= 1 && return (k * dt, 1)
    end
    return (max_steps * dt, 0)
end