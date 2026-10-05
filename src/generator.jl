"""
    comp_generator(V, D, β, N)

Negative generator of the jump process discretizing the overdamped Langevin dynamics with
potential `V`, diffusion `D` and inverse temperature `β`, on `N` interior nodes of a
regular grid on (0, 1), with homogeneous Dirichlet boundary conditions.

```math
L_{ij} = -\\sigma(\\beta,dx^2)\\frac{D_i + D_j}e^{-β (V_j - V_i) / 2}, \\quad |i - j| = 1,\\quad \\sigma(\\beta,dx^2)=(\\beta * dx^2)^{-1}
```
"""
function comp_generator(V, D, β, N)
    X = range(0, 1, N + 2)
    dx = inv(N + 1)

    Vx = V.(X)
    Dx = D.(X)

    @inline prev(i) = (i > 1) ? i - 1 : i
    @inline next(i) = (i < N + 2) ? i + 1 : i

    rate(i, j) = exp(-β * (Vx[j] - Vx[i]) / 2) * (Dx[i] + Dx[j]) / 2
    rows = vcat(([i, i, i] for i in 1:(N + 2))...)
    cols = vcat(([i, prev(i), next(i)] for i in 1:(N + 2))...)

    factor = inv(β * dx^2)
    vals = vcat((factor * [rate(i, prev(i)) + rate(i, next(i)), -rate(i, prev(i)), -rate(i, next(i))]
                 for i in 1:(N + 2))...)

    L = sparse(rows, cols, vals)

    return L[2:(N + 1), 2:(N + 1)]                         # sparse N x N , remove cemetery nodes ⟺ Dirichlet b.c.
end

"""
    comp_qsd(L)

QSD of the killed jump process with negative generator `L`, as a probability vector on the `L`-grid
"""
function comp_qsd(L)
    λs, us = eigs(L'; sigma = 0.0, nev = 2, which = :LR, tol = eps()) # solve left-eigenproblem
    λs = real.(λs)
    us = real.(us)
    ν = abs.(us[:, 1])
    ν /= sum(ν)

    @assert 0 < λs[1] < λs[2]

    return ν
end

"""
    conv_tv(P, ν, ix, tv_tol; max_iter = 1000)

For the positive linear propagator `P`, returns the number of steps for which the normalized `P`-flow converges to `ν` below `tv_tol`
in total variation, starting from a Dirac at node `ix` (or else -1 if `max_iter` is reached)
"""
function conv_tv(P, ν, ix, tv_tol; max_iter = 1000)
    N = size(P, 1)
    l = zeros(N)
    l[ix] = 1

    k = 0
    while true
        sum(abs, l - ν) < tv_tol && return k

        l = P'l # propagate
        l /= sum(l) # normalize
        k += 1

        (k > max_iter) && return -1
    end
end

"""
    project_qsd(ν, Ngrid, nbins)

`nbins` histogram-featurization of the measure `ν` (the QSD in this work) with mass on `Ngrid` interior nodes of (0,1)
"""
function project_qsd(ν, Ngrid, nbins)
    out = zeros(nbins)
    dx = 1.0 / (Ngrid + 1)
    for j in 1:Ngrid
        out[get_bin(j * dx, 0.0, 1.0, nbins)] += ν[j]
    end
    s = sum(out)
    s > 0 && (out ./= s)
    return out
end
