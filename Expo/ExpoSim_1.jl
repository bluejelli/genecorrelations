
using Catalyst, DifferentialEquations.JumpProcesses, DifferentialEquations.EnsembleAnalysis, JLD2

#=============== COLOR SCHEMES =#

exp_chain = @reaction_network begin
    α, 0 --> A
    β, A --> B
    γ, B --> 0
end


# Initial Conditions/ Parameters ==============================================
begin
    t0 = 0.; tf = 200.
    
    α = 1.0; β = 0.1; γ = 0.001;
    α2 = 1.0; β2 = 0.1; γ2 = 0.01;
    α3 = 1.0; β3 = 0.1; γ3 = 0.1;
    α4 = 1.0; β4 = 0.1; γ4 = 0.1;
    α5 = 1.0; β5 = 0.1; γ5 = 1.0;
    α6 = 1.0; β6 = 0.1; γ6 = 10.0;
    α7 = 1.0; β7 = 0.1; γ7 = 100.0;
    α8 = 1.0; β8 = 0.1; γ8 = 1000.0;

    n = 5e4;
    ϕ_0 = 100.0; ψ_0 = 100.0;

    u0 = [:A => ϕ_0, :B => ψ_0]
    tspan = (t0, tf)

    ps1 = [:α => α, :β => β, :γ => γ]
    ps2 = [:α => α2, :β => β2, :γ => γ2]
    ps3 = [:α => α3, :β => β3, :γ => γ3]
    ps4 = [:α => α4, :β => β4, :γ => γ4]
    ps5 = [:α => α5, :β => β5, :γ => γ5]
    ps6 = [:α => α6, :β => β6, :γ => γ6]
    ps7 = [:α => α7, :β => β7, :γ => γ7]
    ps8 = [:α => α8, :β => β8, :γ => γ8]

    exp_jinput_1 = JumpInputs(exp_chain, u0, tspan, ps1)
end



# Solve the model =======================================================================
begin
    exp_jprob_1 = JumpProblem(exp_jinput_1)
    exp_ens_1 = EnsembleProblem(exp_jprob_1)
end

begin
    exp_sol_1 = solve(exp_ens_1, SSAStepper(), EnsembleThreads();trajectories=n)
end

# Mean Level Level O(\Omega^{1/2}) level expansion.
begin
    ts = t0:5.0:tf   # assume the fastest time scale is 1 time unit.
    cov_out_1 = timeseries_point_meancov(exp_sol_1, ts, ts)

    ts2= t0:1.0:tf
    all_As_1 = [componentwise_vectors_timepoint(exp_sol_1, i)[1] for i in collect(ts2)]
    all_Bs_1 = [componentwise_vectors_timepoint(exp_sol_1, i)[2] for i in collect(ts2)]
end

begin
    jldsave("ExpoSim_1_raw_full.jld2"; ts, cov_out_1, ts2, all_As_1, all_Bs_1)
end
