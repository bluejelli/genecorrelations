using DifferentialEquations, GLMakie, LinearAlgebra

using Catalyst, DifferentialEquations.JumpProcesses, DifferentialEquations.EnsembleAnalysis

#=============== COLOR SCHEMES =#
begin
    using Colors, LaTeXStrings

    begin


        flame = Colors.RGB(((228, 87, 46) ./ 255)...)
        delf_blue = Colors.RGB(((41, 51, 92) ./ 255)...)
        orange_web = Colors.RGB(((243, 167, 18) ./ 255)...)
        olivine = Colors.RGB(((168, 198, 134) ./ 255)...)
        air_superiority = Colors.RGB(((102, 155, 188) ./ 255)...)

        saffron = Colors.RGB(((250, 199, 62) ./ 255)...)
        vista = Colors.RGB(((131, 144, 250) ./ 255)...)
        pink = Colors.RGB(((248, 141, 173) ./ 255)...)
        delft = Colors.RGB(((29, 47, 111) ./ 255)...)

        prussian_blue = Colors.RGB(((33, 45, 64) ./ 255)...)
        citron = Colors.RGB(((187, 190, 100) ./ 255)...)
        bittersweet = Colors.RGB(((237, 106, 90) ./ 255)...)
        periwinkle = Colors.RGB(((187, 189, 246) ./ 255)...)

        fontcolor = Colors.RGB(((17, 21, 28) ./ 255)...)
    end
end

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
    exp_jinput_2 = JumpInputs(exp_chain, u0, tspan, ps2)
    exp_jinput_3 = JumpInputs(exp_chain, u0, tspan, ps3)
    exp_jinput_4 = JumpInputs(exp_chain, u0, tspan, ps4)
    exp_jinput_5 = JumpInputs(exp_chain, u0, tspan, ps5)
    exp_jinput_6 = JumpInputs(exp_chain, u0, tspan, ps6)
    exp_jinput_7 = JumpInputs(exp_chain, u0, tspan, ps7)
    exp_jinput_8 = JumpInputs(exp_chain, u0, tspan, ps8)
end

function thvar_B(t, α, β, γ, ψ0, ϕ0)
    # Compute common terms that don't depend on t
    denominator = γ * (β - γ)^2
    
    # Vectorized computation
    exp_2t_beta_gamma = @. exp(-2 * t * (β + γ))
    exp_t_2beta_gamma = @. exp(t * (2 * β + γ))
    exp_t_beta_2gamma = @. exp(t * (β + 2 * γ))
    exp_2t_beta_gamma_pos = @. exp(2 * t * (β + γ))
    exp_2beta_t = @. exp(2 * β * t)
    exp_2gamma_t = @. exp(2 * γ * t)
    exp_t_beta_gamma = @. exp(t * (β + γ))
    
    # Break down the terms for clarity
    term1 = @. (β - γ) * exp_t_2beta_gamma * (γ * (β * (ψ0 + ϕ0) - γ * ψ0) - α * β)
    term2 = @. γ * (γ - β) * (β * ϕ0 - α) * exp_t_beta_2gamma
    term3 = @. α * (β - γ)^2 * exp_2t_beta_gamma_pos
    term4 = @. -γ * exp_2beta_t * (β^2 * ϕ0 + ψ0 * (β - γ)^2)
    term5 = @. -β^2 * γ * ϕ0 * exp_2gamma_t
    term6 = @. 2 * β^2 * γ * ϕ0 * exp_t_beta_gamma
    
    # Combine all terms
    @. (1/denominator) * exp_2t_beta_gamma * (term1 + term2 + term3 + term4 + term5 + term6)
end

function thvar_A(t, α, β, ϕ0)
    # Using the @. macro for vectorized operations
    exp_beta_t = @. exp(β * t)
    exp_neg2beta_t = @. exp(-2 * β * t)
    
    @. exp_neg2beta_t * (exp_beta_t - 1) * (β * ϕ0 + α * exp_beta_t) / β
end


# Solve the model =======================================================================
begin
    exp_jprob_1 = JumpProblem(exp_jinput_1)
    exp_ens_1 = EnsembleProblem(exp_jprob_1)

    exp_jprob_2 = JumpProblem(exp_jinput_2)
    exp_ens_2 = EnsembleProblem(exp_jprob_2)
    
    exp_jprob_3 = JumpProblem(exp_jinput_3)
    exp_ens_3 = EnsembleProblem(exp_jprob_3)

    exp_jprob_4 = JumpProblem(exp_jinput_4)
    exp_ens_4 = EnsembleProblem(exp_jprob_4)
    
    exp_jprob_5 = JumpProblem(exp_jinput_5)
    exp_ens_5 = EnsembleProblem(exp_jprob_5)

    exp_jprob_6 = JumpProblem(exp_jinput_6)
    exp_ens_6 = EnsembleProblem(exp_jprob_6)

    exp_jprob_7 = JumpProblem(exp_jinput_7)
    exp_ens_7 = EnsembleProblem(exp_jprob_7)

    exp_jprob_8 = JumpProblem(exp_jinput_8)
    exp_ens_8 = EnsembleProblem(exp_jprob_8)
end

begin
    exp_sol_1 = solve(exp_ens_1, SSAStepper(), EnsembleThreads();trajectories=n)
    exp_sol_2 = solve(exp_ens_2, SSAStepper(), EnsembleThreads();trajectories=n)
    exp_sol_3 = solve(exp_ens_3, SSAStepper(), EnsembleThreads();trajectories=n)
    exp_sol_4 = solve(exp_ens_4, SSAStepper(), EnsembleThreads();trajectories=n)
    exp_sol_5 = solve(exp_ens_5, SSAStepper(), EnsembleThreads();trajectories=n)
    exp_sol_6 = solve(exp_ens_6, SSAStepper(), EnsembleThreads();trajectories=n)
    exp_sol_7 = solve(exp_ens_7, SSAStepper(), EnsembleThreads();trajectories=n)
    exp_sol_8 = solve(exp_ens_8, SSAStepper(), EnsembleThreads();trajectories=n)
end

function th_crosscovar(ts, β, γ, ϕ0)
    # Vectorize the computation over the time array ts
    return @. -(
        (β * ϕ0 * exp(-2β * ts) * (exp(ts * (β - γ)) - 1)) / (β - γ)
    )
end



# Mean Level Level O(\Omega^{1/2}) level expansion.
begin
    ts = t0:5.0:tf   # assume the fastest time scale is 1 time unit.
    out = timeseries_point_meancov(exp_sol, ts, ts)

    # Get the diagnol of the covariance matrix.
    diag = [out[i, i] for i in 1:size(out)[2]]


    # Extract means
    mean_A =  [diag[I][1][1] for I in eachindex(diag)]
    mean_B =  [diag[I][1][2] for I in eachindex(diag)]

    var_A = [diag[I][3][1] for I in eachindex(diag)]
    var_B = [diag[I][3][2] for I in eachindex(diag)]
  
    
    # Mean Level predictions for A and B
    theory_a = ϕ_0 .* exp.(-β .*ts) .- (α/β)  .* exp.(- β .* ts) .+ (α/β)
    theory_b = ((α * exp.(- β .* ts) .- α .- (β * ϕ_0) .* exp.(-β .* ts))/(β - γ))
    theory_b .+= (α * β) / (γ * (β - γ))
    theory_b .+= ((ψ_0 + ((β * ϕ_0)/(β - γ)) - ((α * β)/(γ * (β - γ)))) .* exp.(- γ .* ts))

end

begin
    fun2 = timepoint_meanvar(exp_sol, ts)
end

begin
    varA = fun2[2][1,:]
    varB = fun2[2][2, :]
    theory_time = t0:5.0:tf
    ts2 = t0:0.5:tf
    thavar = thvar_A(ts2, α, β, ϕ_0)
    thbvar = thvar_B(ts2, α, β, γ, ψ_0, ϕ_0)
end

begin
    var_A = [diag[I][3][1] for I in eachindex(diag)]
    var_B = [diag[I][3][2] for I in eachindex(diag)]
end

begin
    joint_means = zeros(200)
    meansA=[];meansB=[]; varsA=[];varsB=[];
    for i in eachindex(joint_means)
        joint_means[i] = dot(componentwise_vectors_timepoint(exp_sol, i)[1], componentwise_vectors_timepoint(exp_sol, i)[2])
        joint_means[i] /= n
        push!(meansA, componentwise_vectors_timepoint(exp_sol, i)[1])
        push!(meansB, componentwise_vectors_timepoint(exp_sol, i)[2])
    end
end

using Statistics
begin
    crosscovar = [(joint_means[i] - mean(meansA[i]) * mean(meansB[i])) for i in 1:length(meansA)]
end

begin
    stdA = [std(meansA[i]) for i in 1:length(meansA)]
    stdB = [std(meansB[i]) for i in 1:length(meansB)]
end

begin
    correlation_coeff_sim = crosscovar ./ (stdA .* stdB)
end



using CairoMakie
# ==================================== O(V^{0}) ==================================================================== #

begin
    fig = Figure(fonts=(; sci="CMU Serif"), size = (1050, 500), textcolor=fontcolor)

    plot1 = fig[1, 2]
    plot2 = fig[1, 1]

    b_x_label = L"$t/\tau$"
    b_y_label = L" \langle B(t)^2  \rangle - \langle B(t) \rangle^2"

    a_x_label = L"$t/\tau$"
    a_y_label = L" \langle A(t)^2  \rangle - \langle A(t) \rangle^2"

    ax = CairoMakie.Axis(plot1,
        xlabel=b_x_label, ylabel=b_y_label, xlabelsize=30, ylabelsize=30, xticklabelsize=20, yticklabelsize=20, title=L"\text{ }",
        xticklabelfont=:sci, yticklabelfont=:sci, titlesize=27, aspect = 1, xtickcolor= fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
        bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, titlecolor=fontcolor, xlabelcolor =fontcolor,
        ylabelcolor =fontcolor, xgridvisible = false, ygridvisible = false, ytickformat = values -> [(LaTeXString(string(Int(value)))) for value in values], xtickformat = values -> [(LaTeXString(string((value)))) for value in values],)


    th1 = CairoMakie.lines!(ax, collect(ts2) ./ (α + β + γ), thbvar,  linewidth=0.3, color=prussian_blue)
    th2 = CairoMakie.lines!(ax, collect(theory_time) ./ (α + β + γ), theory_b,  linewidth=0.3, color=citron)
    #sm1 = CairoMakie.scatter!(ax, collect(ts[1:end-1]) ./ (α + β + γ), var_B, markersize=5, strokewidth=1.0, strokecolor = air_superiority, color=:transparent)
    sm3 = CairoMakie.scatter!(ax, collect(t0:5.0:tf) ./ (α + β + γ), var_B, markersize=5, strokewidth=1.3, strokecolor = prussian_blue, color=:transparent)
    sm2 = CairoMakie.scatter!(ax, collect(ts) ./ (α + β + γ), mean_B, markersize=5, strokewidth=1.3, strokecolor = citron, color=:transparent)
    xlims!(-1, 100)

    CairoMakie.axislegend(ax, [th1, th2, sm3, sm2],[L"\text{Variance - Theory}", L"\text{Mean - Theory}", L"\text{Variance - Simulation}", L"\text{Mean - Simulation}"],backgroundcolor=:transparent,  position=:rc, xtickcolor=fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
    bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, textcolor=fontcolor, linecolor=fontcolor, framecolor=fontcolor,framevisible=false)

    ax2 = CairoMakie.Axis(plot2,
        xlabel=a_x_label, ylabel=a_y_label, xlabelsize=30, ylabelsize=30, xticklabelsize=20, yticklabelsize=20, title=L"\text{ }",
        xticklabelfont=:sci, yticklabelfont=:sci, titlesize=27, aspect = 1, xtickcolor= fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
        bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, titlecolor=fontcolor, xlabelcolor =fontcolor,
        ylabelcolor =fontcolor, xgridvisible = false, ygridvisible = false, ytickformat = values -> [(LaTeXString(string(Int(value)))) for value in values], xtickformat = values -> [(LaTeXString(string((value)))) for value in values],)


    a_th1 = CairoMakie.lines!(ax2, collect(ts2) ./ (α + β + γ), thavar,  linewidth=0.3, color=bittersweet)
    a_th2 = CairoMakie.lines!(ax2, collect(theory_time) ./ (α + β + γ), theory_a,  linewidth=0.3, color=periwinkle)
    a_sm3 = CairoMakie.scatter!(ax2, collect(t0:5.0:tf) ./ (α + β + γ), var_A, markersize=5, strokewidth=1.3, strokecolor = bittersweet, color=:transparent)
    a_sm2 = CairoMakie.scatter!(ax2, collect(ts) ./ (α + β + γ), mean_A, markersize=5, strokewidth=1.3, strokecolor = periwinkle, color=:transparent)
    xlims!(-1, 100)

    CairoMakie.axislegend(ax2, [a_th1, a_th2, a_sm3, a_sm2],[L"\text{Variance - Theory}", L"\text{Mean - Theory}", L"\text{Variance - Simulation}", L"\text{Mean - Simulation}"],backgroundcolor=:transparent,  position=:rc, xtickcolor=fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
    bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, textcolor=fontcolor, linecolor=fontcolor, framecolor=fontcolor,framevisible=false)


    save("exp_var.png", fig)
end

# Find the difference between simulation and theory to ensure that the simulation is dominated by error due to sample-size

using JLD2


#Cross-covariance


begin
    thcrosscovar = th_crosscovar(ts2, β, γ, ϕ_0)
end

begin
    th_corr_coeff = thcrosscovar ./ (sqrt.(thavar) .* sqrt.(thbvar))
end


# Cross-covariance
begin
    fig = Figure(fonts=(; sci="CMU Serif"), size = (700, 500), textcolor=fontcolor)

    plot1 = fig[1, 1]

    b_x_label = L"$t/\tau$"
    b_y_label = L" \langle B(t)^2  \rangle - \langle B(t) \rangle^2"

    a_x_label = L"$t/\tau$"
    a_y_label = L"\langle \langle \, A(t) B(t) \, \rangle \rangle"

    ax2 = CairoMakie.Axis(plot1,
        xlabel=a_x_label, ylabel=a_y_label, xlabelsize=30, ylabelsize=30, xticklabelsize=20, yticklabelsize=20,
        xticklabelfont=:sci, yticklabelfont=:sci, titlesize=27, aspect = 1, xtickcolor= fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
        bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, titlecolor=fontcolor, xlabelcolor =fontcolor,
        ylabelcolor =fontcolor, xgridvisible = false, ygridvisible = false, ytickformat = values -> [(LaTeXString(string(Int(value)))) for value in values], xtickformat = values -> [(LaTeXString(string(Int(value)))) for value in values])

    cross_covar_sm_1 = CairoMakie.scatter!(ax2, collect(1.0:1.0:tf) ./ (α + β + γ), crosscovar, markersize=5, strokewidth=1.3, strokecolor = pink, color=:transparent)
    cross_covar_th_1 = CairoMakie.lines!(ax2, collect(ts2) ./ (α + β + γ), thcrosscovar,  linewidth=0.3, color=pink)
    xlims!(-1, 100)

    CairoMakie.axislegend(ax2, [cross_covar_sm_1, cross_covar_th_1],[L"\text{Simulation}", L"\text{Theory}"],backgroundcolor=:transparent,  position=:rc, xtickcolor=fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
    bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, textcolor=fontcolor, linecolor=fontcolor, framecolor=fontcolor,framevisible=false)

    save("exp_cross_covar.png", fig)
end


# Correlation Coefficient
begin
    fig = Figure(fonts=(; sci="CMU Serif"), size = (700, 500), textcolor=fontcolor)

    plot1 = fig[1, 1]

    b_x_label = L"$t/\tau$"
    b_y_label = L" \langle B(t)^2  \rangle - \langle B(t) \rangle^2"

    a_x_label = L"$t/\tau$"
    a_y_label = L" \rho_{AB}(t)"

    ax2 = CairoMakie.Axis(plot1,
        xlabel=a_x_label, ylabel=a_y_label, xlabelsize=30, ylabelsize=30, xticklabelsize=20, yticklabelsize=20, title=L"\text{ }",
        xticklabelfont=:sci, yticklabelfont=:sci, titlesize=27, aspect = 1, xtickcolor= fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
        bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, titlecolor=fontcolor, xlabelcolor =fontcolor,
        ylabelcolor =fontcolor, xgridvisible = false, ygridvisible = false, ytickformat = values -> [(LaTeXString(string(round((value), digits=2)))) for value in values], xtickformat = values -> [(LaTeXString(string((value)))) for value in values],)


    ρsm = CairoMakie.scatter!(ax2, collect(1.0:1.0:tf) ./ (α + β + γ), correlation_coeff_sim, markersize=5, strokewidth=1.3, strokecolor = orange_web, color=:transparent)
    ρth = CairoMakie.lines!(ax2, collect(ts2) ./ (α + β + γ), th_corr_coeff,  linewidth=0.3, color=orange_web)
    xlims!(-1, 100)

    CairoMakie.axislegend(ax2, [ρsm, ρth],[L"\text{Simulation}", L"\text{Theory}"],backgroundcolor=:transparent,  position=:rc, xtickcolor=fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
    bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, textcolor=fontcolor, linecolor=fontcolor, framecolor=fontcolor,framevisible=false)


    save("exp_correlation_coefficient.png", fig)
end

# Fano Factor

begin
    ts = t0:0.5:tf
    theory_a_f = ϕ_0 .* exp.(-β .*ts) .- (α/β)  .* exp.(- β .* ts) .+ (α/β)
    theory_b_f = ((α * exp.(- β .* ts) .- α .- (β * ϕ_0) .* exp.(-β .* ts))/(β - γ))
    theory_b_f .+= (α * β) / (γ * (β - γ))
    theory_b_f .+= ((ψ_0 + ((β * ϕ_0)/(β - γ)) - ((α * β)/(γ * (β - γ)))) .* exp.(- γ .* ts))
end
begin
    a_fano_factor = var_A ./ mean_A
    b_fano_factor = var_B ./ mean_B
end

begin
    a_th_fano_factor = thavar ./ theory_a_f
    b_th_fano_factor = thbvar ./ theory_b_f
end


begin
    fig = Figure(fonts=(; sci="CMU Serif"), size = (700, 500), textcolor=fontcolor)

    plot1 = fig[1, 1]

    b_x_label = L"$t/\tau$"
    b_y_label = L" \langle B(t)^2  \rangle - \langle B(t) \rangle^2"

    a_x_label = L"$t/\tau$"
    a_y_label = L"F(t)"

    ax2 = CairoMakie.Axis(plot1,
        xlabel=a_x_label, ylabel=a_y_label, xlabelsize=30, ylabelsize=30, xticklabelsize=20, yticklabelsize=20, title=L"\text{ }",
        xticklabelfont=:sci, yticklabelfont=:sci, titlesize=27, aspect = 1, xtickcolor= fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
        bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, titlecolor=fontcolor, xlabelcolor =fontcolor,
        ylabelcolor =fontcolor, xgridvisible = false, ygridvisible = false, ytickformat = values -> [(LaTeXString(string(round((value), digits=2)))) for value in values], xtickformat = values -> [(LaTeXString(string((value)))) for value in values],)


    a_fano_sim = CairoMakie.scatter!(ax2, collect(t0:5.0:tf) ./ (α + β + γ), a_fano_factor, markersize=5, strokewidth=1.3, strokecolor = delft, color=:transparent)
    a_fano_th = CairoMakie.lines!(ax2, collect(ts) ./ (α + β + γ), a_th_fano_factor,  linewidth=0.3, color=delft)

    b_fano_sim = CairoMakie.scatter!(ax2, collect(t0:5.0:tf) ./ (α + β + γ), b_fano_factor, markersize=5, strokewidth=1.3, strokecolor = bittersweet, color=:transparent)
    b_fano_th = CairoMakie.lines!(ax2, collect(ts) ./ (α + β + γ), b_th_fano_factor,  linewidth=0.3, color=bittersweet)

    xlims!(-1, 100)

    CairoMakie.axislegend(ax2, [a_fano_sim, a_fano_th, b_fano_sim, b_fano_th],[L"F_{B}(t) - \text{Simulation}",L"F_{B}(t) - \text{Theory}", L"F_{A}(t) - \text{Simulation}",L"F_{A}(t) - \text{Theory}",],backgroundcolor=:transparent,  position=:rc, xtickcolor=fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
    bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, textcolor=fontcolor, linecolor=fontcolor, framecolor=fontcolor,framevisible=false)


    save("exp_fano_factor.png", fig)
end

jldsave("exp_k+2.jld2"; out, mean_A, mean_B, var_A, var_B)
