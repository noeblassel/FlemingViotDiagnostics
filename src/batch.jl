# Author: Noe Blassel
# Last modified: Oct 5 2026

const _PAD = -1.0f0 # padding value for out-of-sequence elements

"""
    draw_system(rng; βlims, Ngrid, tau_gt, potential_kwargs)

Generates one sample `(U, D, β)` from the prior, compute its QSD `ν` on a grid of `Ngrid` nodes and the
killed semigroup propagator `P_gt` using the lag time `tau_gt`.

Keyword arguments:
- `βlims` : range of inverse temperatures
- `Ngrid` : size of discretization grid (for QSD and semigroup discretization)
- `tau_gt` : lag parameter for reference decorrelation times
- `potential_kwargs`: prior specification, passed to `generate_potential`

Return fields
- `W′` : potential derivative (callable)
- `D` : diffusion profile (callable)
- `D′` : its derivative (callable)
- `β` : inverse temperature (`Float64`)
- `ν` : QSD (Vector{Float64}, of size `Ngrid`)
- `P_gt` : semigroup propagator (`Matrix{Float64}`, of size `Ngrid`×`Ngrid`)
"""
function draw_system(rng; βlims, Ngrid, tau_gt, potential_kwargs = NamedTuple())
    W, D, W′, D′ = generate_potential(; rng = rng, potential_kwargs...)
    βmin, βmax = βlims
    β = βmin + (βmax - βmin) * rand(rng)
    L = comp_generator(W, D, β, Ngrid)
    ν = comp_qsd(L)
    P_gt = exp(-tau_gt * Matrix(L))
    return (W′ = W′, D = D, D′ = D′, β = β, ν = ν, P_gt = P_gt)
end

"""
    simulate_traces(rng, sys; ntrace, stride_lims, Nreplicas_lims, Ngrid, dt, tol, tau_gt,
                    min_length, ncorr, max_attempts)

Simulates `ntrace` Fleming-Viot processes, with initial points drawn uniformly in (0,1),
under overdamped Langevin dynamics specified by `sys` (as returned by `draw_system`).

Keyword arguments:
- `stride_lims` : range of strides between frames. Design lag τ = `stride_lims * dt`
- `Nreplicas_lims` : range for the (random) number of replicas
- `Ngrid` : size of discretization grid (for semigroup propagation)
- `dt` : time step in the Euler-Maruyama scheme
- `tol` : total variation tolerance parameter (εcorr in the paper)
- `tau_gt` : lag parameter for reference decorrelation times
- `min_length` : minimum number of frames per trace
- `ncorr` : length of a simulation, in units of the reference decorrelation time
- `max_attempts` : max number of attempts before a potential is skipped

Returns a vector of frames, each with fields
- `fv_frames` : the histogram trace
- `gr_history` : the trace of the gelman-rubin statistic (minus one)
- `Nreplicas` : the number of replicas
- `stride` : the lag between frames, in units of `dt`
- `T_conv` : the honest decorrelation time
- `success` : a boolean recording whether traces were sampled

"""
function simulate_traces(rng, sys; ntrace, stride_lims, Nreplicas_lims, Ngrid, dt, tol, tau_gt,
                         min_length, ncorr, max_attempts)
    trace_seeds = rand(rng, UInt64, ntrace)
    trace_strides = rand(rng, stride_lims[1]:stride_lims[2], ntrace)
    trace_Nreplicas = rand(rng, Nreplicas_lims[1]:Nreplicas_lims[2], ntrace)
    trace_ixs = rand(rng, 1:Ngrid, ntrace)

    results = Vector{Any}(undef, ntrace)

    Threads.@threads for j in 1:ntrace
        local_rng = Random.Xoshiro(trace_seeds[j])
        stride_j = trace_strides[j]
        Nreplicas_j = trace_Nreplicas[j]
        ix_j = trace_ixs[j]

        local_failed = 0
        success = false
        T_conv = 0.0
        local_fv = nothing

        while !success
            try
                x0 = (ix_j + 1) / (Ngrid + 2)
                k_gt = conv_tv(sys.P_gt, sys.ν, ix_j, tol)
                k_gt < 0 && throw(ErrorException("ground truth failed to converge"))
                T_conv = max(2, k_gt) * tau_gt # honest decorrelation time
                nframes_target = max(min_length, round(Int, ncorr * T_conv / (stride_j * dt)))
                local_fv = sim_fv(sys.W′, sys.D, sys.D′, dt, sys.β, Nreplicas_j,
                                  nframes_target * stride_j, stride_j, x0, local_rng)

                if length(local_fv.fv_frames) >= min_length
                    success = true
                else
                    local_failed += 1
                end
            catch e
                local_failed += 1
                @warn "Fleming-Viot run failed" exception = (e, catch_backtrace())
            end

            if local_failed >= max_attempts
                @info "Exceeded maximum attempts ($max_attempts) for trace $j, skipping"
                break
            end

            ix_j = rand(local_rng, 1:Ngrid)
        end

        results[j] = success ?
                     (fv_frames = local_fv.fv_frames, gr_history = local_fv.gr_history,
                      Nreplicas = Nreplicas_j, stride = stride_j, T_conv = T_conv, success = true) :
                     (success = false,)
    end

    return results
end

"""
    get_batch(rng; tol = 0.05, Ngrid = 100, βlims = (1.0, 3.0), dt = 1e-3, tau_gt = 0.1,
              stride_lims = (10, 200), Nreplicas_lims = (10, 200), input_dim = 64, ntrace = 5,
              ncut = 1, npot = 5, min_length = 5, ncorr = 2, max_attempts = 10,
              potential_kwargs = NamedTuple())

Assembles a batch of labeled feature sequences

Keyword arguments:
- `tol`: total variation tolerance parameter (εcorr in the paper)
- `Ngrid`: size of discretization grid (for semigroup propagation)
- `βlims` : range of inverse temperatures
- `dt` : time step in the Euler-Maruyama scheme
- `tau_gt`: lag parameter for reference decorrelation times
- `stride_lims` : range of strides between frames. Design lag τ = `stride_lims * dt`
- `Nreplicas_lims` : range for the (random) number of replicas
- `input_dim`: number of histogram bins
- `ntrace`: number of Fleming-Viot simulations per system
- `ncut`: number of random cuts per trace (samples of `L`) between the decorrelation frame and the last frame.` ncut=0` computes whole feature sequences
- `npot`: number of systems sampled per batch
- `min_length`: mininimum number of frames from a simulation
- `ncorr`: maximum length of Fleming--Viot trajectories (units of the reference)
- `potential_kwargs`: prior specification, passed to `generate_potential`
- `ncorr` : length of a simulation, in units of the reference decorrelation time
- `max_attempts` : max number of attempts before a potential is skipped
- `Ngrid` : size of discretization grid (for QSD and semigroup discretization)
- `tau_gt` : lag parameter for reference decorrelation times
- `potential_kwargs`: prior specification, passed to `generate_potential`

Returns `(X, Y, mask, B, T_conv)`:
- `X`: the feature batch (`Array{Float32}` of size `(input_dim+1,length,batch)`), with appended design features
- `Y`: label batch (`Array{Float32}` of size `(length,batch)`)
- `mask`: (`Array{Bool}` recording within-sequence indices (batches contain different-length sequences)
- `B`: (`Array{Float32}` of size `(2, length, batch)`) giving baselines (Gelman-Rubin -1 , total variation of histogram frames to reference QSD)
- `T_conv`: reference decorrelation time
"""
function get_batch(rng;
        tol = 0.05,
        Ngrid = 100,
        βlims = (1.0, 3.0),
        dt = 1e-3,
        tau_gt::Float64 = 0.1,
        stride_lims::Tuple{Int, Int} = (10, 200),
        Nreplicas_lims::Tuple{Int, Int} = (10, 200),
        input_dim = 64,
        ntrace = 5,
        ncut = 1,
        npot = 5,
        min_length = 5,
        ncorr = 2,
        max_attempts::Int = 10,
        potential_kwargs::NamedTuple = NamedTuple())
    batch = Vector{Vector{Float32}}[]
    baselines = Vector{Vector{Float32}}[]
    labels = Vector{Float32}[]
    tconvs = Float64[]

    i = 0
    while i < npot
        sys = draw_system(rng; βlims, Ngrid, tau_gt, potential_kwargs)
        ν_binned = project_qsd(sys.ν, Ngrid, input_dim)
        traces = simulate_traces(rng, sys; ntrace, stride_lims, Nreplicas_lims, Ngrid, dt, tol,
                                 tau_gt, min_length, ncorr, max_attempts)
        any(r -> !r.success, traces) && continue

        for r in traces
            l = length(r.fv_frames)
            meta = meta_scalar(r.Nreplicas, r.stride, dt)
            features = [vcat(hist_feature(f, input_dim), [meta]) for f in r.fv_frames]
            base = [Float32.([r.gr_history[k], tv_from_qsd(r.fv_frames[k], ν_binned, input_dim)])
                    for k in 1:l]
            full_labels = (1:l) .* (r.stride * dt) .> r.T_conv

            for _ in 1:ncut
                decorr_frame = max(min_length, ceil(Int, r.T_conv / (r.stride * dt)))
                len = clamp(rand(rng, decorr_frame:l), min_length, l)
                push!(batch, features[1:len])
                push!(baselines, base[1:len])
                push!(labels, Float32.(full_labels[1:len]))
                push!(tconvs, r.T_conv)
            end

            if ncut == 0
                push!(batch, features)
                push!(baselines, base)
                push!(labels, Float32.(full_labels))
                push!(tconvs, r.T_conv)
            end
        end

        i += 1
    end

    nsamples = length(batch)
    nsamples == 0 && error("No samples collected. Increase npot, ntrace or max_attempts")
    p = randperm(rng, nsamples)

    batch_X = batchseq(view(batch, p), zeros(Float32, input_dim + 1))
    batch_B = batchseq(view(baselines, p), zeros(Float32, 2))
    batch_Y = batchseq(view(labels, p), _PAD)

    Y = stack(batch_Y; dims = 1)
    mask = (Y .!= _PAD)

    return (stack(batch_X; dims = 2), Y, mask, stack(batch_B; dims = 2), tconvs[p])
end
