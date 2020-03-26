# -----------------------------------------------------------------------------
# GIBBS SAMPLING
# Econometrics IV, Part 1
# Prof. Frank Schorfheide
#
# PS2 - Exercise 2
# -----------------------------------------------------------------------------

# ---------
# Packages
# ---------
using Distributions, Random
using Statistics
using Plots, LaTeXStrings

# --------
# Modules
# --------
include("GS.jl")
using .GS

# ----------------
# 1. Housekeeping
# ----------------
# Monte Carlo sample
N = 100                         # Total number of draws
N_burn = Int(0.5 * N)           # Discarted draws
N_run = 50                      # Number of runs

# Univariate means
μ_1 = 0
μ_2 = 0

# Variance-covariance matrix
σ_1 = 1                         # Standard deviation of Y1
σ_2 = 1                         # Standard deviation of Y2
ρ = collect(-0.95:0.1:0.95)     # Correlation coefficient


# -------------
# 2. Algorithm
# -------------

#  Pre-allocation
nAlternatives = length(ρ)

tY1 = zeros(N - N_burn, N_run, nAlternatives)
mY1Mean = zeros(N_run, nAlternatives)
mY1Bias = zeros(N_run, nAlternatives)
mY1Var = zeros(N_run, nAlternatives)
mY1RMSE = zeros(N_run, nAlternatives)

tY2 = similar(tY1)
mY2Mean = similar(mY1Mean)
mY2Bias = similar(mY1Bias)
mY2Var = similar(mY1Var)
mY2RMSE = similar(mY1RMSE)

# For deterministic (repicable) results
Random.seed!(123)

for iAlternative in 1:nAlternatives

    for iRun in 1:N_run

        # Initialization
        μ_0 = 1
        σ_0 = 1
        d_0 = Normal(μ_0, σ_0)
        Y2_0 = rand(d_0, 1)[1]

        for i in 1:N

            # Sampler
            Y1, Y2 = GS.gibbs(μ_1, σ_1, μ_2, σ_2, ρ[iAlternative], Y2_0)

            # Update initial condition
            Y2_0 = Y2

            # Save results
            if  i > N_burn
                j = i - N_burn
                tY1[j, iRun, iAlternative] = Y1
                tY2[j, iRun, iAlternative] = Y2
            end

        end

        # Mean, bias and variance
        # Y1
        mY1Mean[iRun, iAlternative] = mean(tY1[:,iRun, iAlternative])
        mY1Bias[iRun, iAlternative] = (mY1Mean[iRun, iAlternative] - μ_1)
        mY1Var[iRun, iAlternative] = var(tY1[:,iRun, iAlternative])
        mY1RMSE[iRun, iAlternative] = sqrt( mean( (tY1[:,iRun, iAlternative].-μ_1).^2 ) )

        # Y2
        mY2Mean[iRun, iAlternative] = mean(tY2[:,iRun, iAlternative])
        mY2Bias[iRun, iAlternative] = (mY2Mean[iRun, iAlternative] - μ_2)
        mY2Var[iRun, iAlternative] = var(tY2[:,iRun, iAlternative])
        mY2RMSE[iRun, iAlternative] = sqrt( mean( (tY2[:,iRun, iAlternative].-μ_2).^2 ) )

    end
end

# ---------
# Graphs
# ---------

# Bias
pBias = plot(ρ, mean(mY1Bias, dims=1)', xlabel=L"\rho", marker=:o, label=L"Y_1")
plot!(ρ, mean(mY2Bias, dims=1)', marker=:o, label=L"Y_2")

savefig(pBias, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV - Part I/PS/PS2/LaTeX/pBias.pdf")


# Variance
pVariance = plot(ρ, mean(mY1Var, dims=1)', xlabel=L"\rho", marker=:o, label=L"Y_1", legend=:bottomright)
plot!(ρ, mean(mY2Var, dims=1)', marker=:o, label=L"Y_2")

savefig(pVariance, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV - Part I/PS/PS2/LaTeX/pVariance.pdf")


# RMSE
pRMSE = plot(ρ, mean(mY1RMSE, dims=1)', xlabel=L"\rho", marker=:o, label=L"Y_1", legend=:topright)
plot!(ρ, mean(mY2RMSE, dims=1)', marker=:o, label=L"Y_2")

savefig(pRMSE, "/Users/Castesil/Documents/EUI/Year II - PENN/Spring 2020/Econometrics IV - Part I/PS/PS2/LaTeX/pRMSE.pdf")
