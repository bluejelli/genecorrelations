using DifferentialEquations, GLMakie, LinearAlgebra

using Catalyst, DifferentialEquations.JumpProcesses, DifferentialEquations.EnsembleAnalysis

hyp_chain = @reaction_network begin
    α, 0 --> A
    β, A --> B
    γ, B --> 0
end


# Initial Conditions/ Parameters ==============================================
begin
    t0 = 0.; tf = 2000.
    α = 1.0; β = 0.01; γ = 0.001;
    n = 1000;
    ϕ_0 = 500; ψ_0 = 500
    u0 = [:A => ϕ_0, :B => ψ_0]
    tspan = (t0, tf)
    ps = [:α => α, :β => β, :γ => γ]
    hyp_jinput = JumpInputs(hyp_chain, u0, tspan, ps)
end


# Solve the model =======================================================================
begin
    hyp_prob = JumpProblem(hyp_jinput)
    hyp_ens = EnsembleProblem(hyp_prob)

    hyp_sol = solve(hyp_ens, SSAStepper(), EnsembleThreads();trajectories=n)
end




# ===================================================== O (V^1/2) ===============================================
begin
    ts = 0.0:1.0:2000.0     # assume the fastest time scale is 1 time unit.
    mean_traj = timeseries_point_mean(hyp_sol, ts)
    # Mean Level predictions for A and B
    
end


lines(collect(mean_traj.t), collect(mean_traj[1, :]))
lines!(collect(mean_traj.t), collect(mean_traj[2, :]))
lines!(collect(ts), theory_a)
lines!(collect(ts), theory_b)


# ==================================== O(V^{0}) ==================================================================== #

begin
    ts = 0.0:12.0:2000.0
    mean_traj, cov = timeseries_point_meancov(hyp_sol, ts)
    
end