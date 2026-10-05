"""
    sim_fv(V′, D, D′, dt, β, Nrep, nsteps, stride, x0, rng)

Fleming–Viot simulation of `Nrep` replicas started at `x0`, evolved by the Euler–Maruyama scheme with timestep `dt` for the overdamped Langevin equation

```math
dX_t = (-D V' + β^{-1} D')(X_t) dt + \\sqrt{2 β^{-1} D(X_t)} dW_t,
```
killed on the boundary of (0, 1).

Returns `(fv_frames, gr_history)`: a sequence of `Nrep * stride` replica positions, and the corresponding Gelman-Rubin trace (minus one).
"""
function sim_fv(V′, D, D′, dt, β, Nrep, nsteps, stride, x0, rng)
    fv_frames = Vector{Float32}[] # Float32 for RNN inference
    fv = fill(x0, Nrep)

    invβ = inv(β)
    σ = sqrt(2invβ * dt)

    fv_trace_buf = Vector{Float64}(undef, Nrep * stride) # Float64 for simulation
    survived_buf = Vector{Bool}(undef, Nrep)
    Dv = Vector{Float64}(undef, Nrep)

    gr_hist = Float64[]

    sum_lin_gr = zeros(Nrep)
    sum_sq_gr = zeros(Nrep)

    for k in 1:nsteps
        @. Dv = D(fv)
        fv .+= (-Dv .* V′.(fv) + invβ * D′.(fv)) * dt + σ * sqrt.(Dv) .* randn(rng, Nrep)

        @. survived_buf = (0 < fv < 1)
        n_survived = count(survived_buf)

        (n_survived == 0) && error("Extinction Event !")

        fv[.!(survived_buf)] .= rand(rng, fv[survived_buf], Nrep - n_survived)

        step_in_window = mod1(k, stride)
        fv_trace_buf[((step_in_window - 1) * Nrep + 1):(step_in_window * Nrep)] .= fv

        sum_lin_gr .+= fv
        sum_sq_gr .+= fv .^ 2

        if k % stride == 0
            push!(gr_hist,
                  (sum(sum_sq_gr) - sum(sum_lin_gr)^2 / (Nrep * k)) /
                  (sum(sum_sq_gr - (sum_lin_gr .^ 2) / k)) - 1)
            push!(fv_frames, copy(fv_trace_buf))
        end
    end

    return (fv_frames = fv_frames, gr_history = gr_hist)
end
