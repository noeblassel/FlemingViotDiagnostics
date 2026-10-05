"""
    rank_auc(scores, labels)

ROC area under the curve from Wilcoxon-Mann-Whitney U statistic
```math
AUC = \\frac{1}{n_0 n_1} \\sum_{x\\in D_0}\\sum_{y\\in D_1} \\left(1_{f(x)<f(y)} + \\frac12 1_{f(x)=f(y)}\\right)
```
for a binary classifier `f` with `n_1` positive examples `D_1` and `n_0` negative examples `D_0`.

The statistic is computed efficiently using sorting. If `s=(f_1,...,f_n)` is an ordered vector of scores, and
`(f_i,...,f_j)` is a subvector of tied scores (with `f_{i-1}<f_i` and `f_j<f_{j+1}`), the mid-rank is `midrank(f_k) = (i+j)/2` for `i<= k <= j`.
If `f_k` is a unique value, `midrank(f_k)=rank(f_k)` is just the rank. The computation relies on the simple algebraic identity
```math
AUC = \\frac1{n_0 n_1} \\left(\\sum_{y\\in D_1} midrank(f(y)) - \\frac{n_1(n_1+1)}{2} \\right)
```
"""
function rank_auc(scores, labels)
    y = labels .== 1
    npos, nneg = count(y), count(!, y)
    (npos == 0 || nneg == 0) && return NaN
    o = sortperm(scores)
    r = zeros(length(scores)) # midranks
    i = 1
    while i <= length(o)
        j = i
        while j < length(o) && scores[o[j + 1]] == scores[o[i]]
            j += 1
        end
        r[o[i:j]] .= (i + j) / 2
        i = j + 1
    end
    return (sum(r[y]) - npos * (npos + 1) / 2) / (npos * nneg) # rank AUC estimator
end

"""
    full_roc(scores, labels; maxpoints = 1000)

Computes the ROC curve `t→(FPR(t), TPR(t))` between (0,0) and (1,1), where `t` is the decision threshold.
The computed curve is subsampled to `maxpoints`.

Using the identity
```math
FPR(t) = \\frac1{n_0}\\sum_{x\\in D_0} 1_{f(x)\\geq t},\\quad \\frac1{n_1}TPR(t) = \\sum_{y\\in D_1} 1_{f(y)\\geq t}
```
computed from sorted score values `t`, then subsampled to at most `maxpoints`
"""
function full_roc(scores, labels; maxpoints::Int = 1000)
    isempty(scores) && return (fpr = Float64[], tpr = Float64[])
    y = labels .== 1
    npos, nneg = max(count(y), 1), max(count(!, y), 1)
    o = sortperm(scores; rev = true)
    fpr = [0.0]
    tpr = [0.0]
    tp = 0
    fp = 0
    for (k, i) in enumerate(o)
        y[i] ? (tp += 1) : (fp += 1)
        if k == length(o) || scores[o[k + 1]] != scores[i]
            push!(fpr, fp / nneg)
            push!(tpr, tp / npos)
        end
    end
    keep = unique(round.(Int, range(1, length(fpr); length = min(maxpoints, length(fpr)))))
    return (fpr = fpr[keep], tpr = tpr[keep])
end
