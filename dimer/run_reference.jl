# Author: Noe Blassel
# Last modified: Nov 17 2026

# Fleming-Viot simulations for reference QSDs

# usage (from repo root):
#   julia --project=dimer -t auto dimer/run_reference.jl --setting=slow --point=1 --runs=1:144 [--out=path]

using JLD2, Random, Statistics
include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "fv.jl"))

function main(args = ARGS)
    setting_name = getarg(args, "setting", "")
    a = setting(setting_name)
    point = getarg(args, "point", 1)
    lo, hi = parse.(Int, split(getarg(args, "runs", "1:$nsamps_ref"), ":"))
    runs = lo:hi
    outf = getarg(args, "out", joinpath(dimer_setting_dir(setting_name), "shards", "ref_p$(point)_$(lpad(lo, 4, '0')).jld2"))

    stride = a.stride_ref
    nframes = round(Int, Tsim_ref / (stride * sim_dt))
    q0 = load(joinpath(DATA, "dimer", "initial_conditions.jld2"), "coords")[point]
    println("reference: point $(initial_config_labels[point]), runs $lo:$hi, N = $N_ref, stride = $stride, $nframes frames, ",
            "d_dimer = $(a.d_dimer), $(Threads.nthreads()) threads")

    nruns = length(runs)
    xi = Array{Float32}(undef, N_ref, nframes, nruns)
    n_coord = Array{Float32}(undef, N_ref, nframes, nruns)
    gr = Array{Float64}(undef, nframes, nruns)
    n_branchings = zeros(Int, nruns)
    fail = fill("", nruns)

    t0 = time()
    Threads.@threads for ri in 1:nruns
        try
            r = run_fv(q0; N = N_ref, stride = stride, nframes = nframes, dt = sim_dt, d_dimer = a.d_dimer,
                       seed = seed_ref(setting_name, point, runs[ri]), nbins = nbins_max, record_hist = false)
            xi[:, :, ri] .= r.xi
            n_coord[:, :, ri] .= r.ncoord
            gr[:, ri] .= r.gr
            n_branchings[ri] = r.n_branch
        catch e
            fail[ri] = sprint(showerror, e)
            xi[:, :, ri] .= NaN32
        end
    end
    println(count(isempty, fail), "/$nruns runs ok in ", round(time() - t0; digits = 1), " s")
    foreach(i -> isempty(fail[i]) || println("  run $(runs[i]) failed: ", first(fail[i], 120)), 1:nruns)

    mkpath(dirname(outf))
    jldsave(outf; point = point, label = initial_config_labels[point], r0 = initial_dimer_lengths[point], runs = collect(runs),
            ok = isempty.(fail), N = N_ref, stride = stride, nframes = nframes, dt = sim_dt, d_dimer = a.d_dimer,
            xi = xi, ncoord = n_coord, gr = gr, n_branch = n_branchings, failures = fail)
    println("wrote $outf")
end

main()
