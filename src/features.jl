# Author: Noe Blassel
# Last modified: Oct 5 2026

@inline get_bin(val, minval, maxval, nbins) =
    1 + clamp(floor(Int, nbins * (val - minval) / (maxval - minval)), 0, nbins - 1) # histogram bin index

"""
    hist_feature(pts, nbins)

Compute histogram for the samples `pts` on the `nbins`-cell partition of [0, 1] normalized so
    ```math
        \\int_0^1 h(x)dx = 1
    ```
    where `h` is the corresponding step function
"""
function hist_feature(pts, nbins)
    X = zeros(Float32, nbins)
    dx = 1.0f0 / nbins
    for p in pts
        X[get_bin(p, 0.0, 1.0, nbins)] += 1.0f0
    end
    X /= (sum(X) * dx)
    return X
end


meta_scalar(N, stride, dt) = Float32(sqrt(N * stride * dt)) # design feature appended to one histogram feature

"""
    tv_from_qsd(frame, ν_binned, nbins)
total variation distance between histogram frame and histogram QSD
"""
function tv_from_qsd(frame, ν_binned, nbins)
    emp = zeros(nbins)
    for p in frame
        emp[get_bin(p, 0.0, 1.0, nbins)] += 1.0
    end
    s = sum(emp)
    s > 0 && (emp ./= s)
    return sum(abs, emp - ν_binned)
end
