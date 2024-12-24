include("C:/Users/anish/Documents/GitHub/genecorrelations/ThemeColors.jl")
include("C:/Users/anish/Documents/GitHub/genecorrelations/Expo/ExpoTheory.jl")
using .ThemeColors
using .ExpoTheory

# Parameter sets
begin
    α = 1.0; β = 0.1; γ = 0.001;
    α2 = 1.0; β2 = 0.1; γ2 = 0.01;
    α3 = 1.0; β3 = 0.1; γ3 = 0.1;
    α4 = 1.0; β4 = 0.1; γ4 = 0.1;
    α5 = 1.0; β5 = 0.1; γ5 = 1.0;
    α6 = 1.0; β6 = 0.1; γ6 = 10.0;
    α7 = 1.0; β7 = 0.1; γ7 = 100.0;
    α8 = 1.0; β8 = 0.1; γ8 = 1000.0;


    ϕ_0 = 100.0; ψ_0 = 100.0;

end

# Load all of the simulations
begin
     sim_1 = jldopen("Expo/ExpoSim_1_raw_full.jld2")
     sim_2 = jldopen("Expo/ExpoSim_2_raw_full.jld2")
     sim_3 = jldopen("Expo/ExpoSim_3_raw_full.jld2")
     sim_4 = jldopen("Expo/ExpoSim_4_raw_full.jld2")
     sim_5 = jldopen("Expo/ExpoSim_5_raw_full.jld2")
     sim_6 = jldopen("Expo/ExpoSim_6_raw_full.jld2")
     sim_7 = jldopen("Expo/ExpoSim_7_raw_full.jld2")
     sim_8 = jldopen("Expo/ExpoSim_8_raw_full.jld2")

     all_simulations = [sim_1, sim_2, sim_3, sim_4, sim_5, sim_6, sim_7, sim_8]
end

# Extract all of the covariance matricies.
begin
    all_covs = [];
    for (i, sim) in enumerate(all_simulations)
        var = "cov_out_" * string(i)
        push!(all_covs, sim[var])
    end
end

using Statistics, LinearAlgebra

# Extract ALL DATA 
begin
    sims_second_moment_data = []

    for (i, sim) in enumerate(all_simulations)
        joint_means = zeros(length(collect(sim["ts2"])))
        
        meansA=[];meansB=[]; varsA=[];varsB=[];

        A_tag = "all_As_" * string(i)
        B_tag = "all_Bs_" * string(i)
        
        for i in eachindex(joint_means)
            joint_means[i] = dot(sim[A_tag][i], sim[B_tag][i])
            joint_means[i] /= n
            push!(meansA, sim[A_tag][i])
            push!(meansB, sim[B_tag][i])
        end
        
        crosscovar = [(joint_means[i] - mean(meansA[i]) * mean(meansB[i])) for i in eachindex(meansA)]
        stdA = [std(meansA[i]) for i in eachindex(meansA)]
        stdB = [std(meansB[i]) for i in eachindex(meansB)]
        correlation_coeff_sim = crosscovar ./ (stdA .* stdB)

        push!(sims_second_moment_data, (crosscovar, correlation_coeff_sim, meansA, meansB, stdA, stdB))


    end
end


begin
    means = []; vars = []
    for out in all_covs

        diag = [out[i, i] for i in 1:size(out)[2]]
        # Extract means
        mean_A =  [diag[I][1][1] for I in eachindex(diag)]
        mean_B =  [diag[I][1][2] for I in eachindex(diag)]
      
        var_A = [diag[I][3][1] for I in eachindex(diag)]
        var_B = [diag[I][3][2] for I in eachindex(diag)]
        
        push!(means, (mean_A, mean_B))
        push!(vars, (var_A, var_B)) 
    end
end

# Theoretical quantities
begin
    sim_time = collect(sim_1["ts"])
    theory_time = sim_time[1]:0.5:sim_time[end]

    # Parameter set 1:
    mean_th_A_1 = th_mean_A(theory_time, α, β, γ, ϕ_0, ψ_0)
    mean_th_B_1 = th_mean_B(theory_time, α, β, γ, ϕ_0, ψ_0)
    var_th_A_1 = th_var_A(theory_time, α, β, ϕ_0)
    var_th_B_1 = th_var_B(theory_time, α, β, γ, ψ_0, ϕ_0)
    cross_covar_th_1 = th_cross_covar(theory_time, β, γ, ϕ_0)

    mean_th_A_2 = th_mean_A(theory_time, α2, β2, γ2, ϕ_0, ψ_0)
    mean_th_B_2 = th_mean_B(theory_time, α2, β2, γ2, ϕ_0, ψ_0)
    var_th_A_2 = th_var_A(theory_time, α2, β2, ϕ_0)
    var_th_B_2 = th_var_B(theory_time, α2, β2, γ2, ψ_0, ϕ_0)
    cross_covar_th_2 = th_cross_covar(theory_time, β2, γ2, ϕ_0)

    mean_th_A_3 = th_mean_A(theory_time, α3, β3, γ3, ϕ_0, ψ_0)
    mean_th_B_3 = th_mean_B(theory_time, α3, β3, γ3, ϕ_0, ψ_0)
    var_th_A_3 = th_var_A(theory_time, α3, β3, ϕ_0)
    var_th_B_3 = th_var_B(theory_time, α3, β3, γ3, ψ_0, ϕ_0)
    cross_covar_th_3 = th_cross_covar(theory_time, β3, γ3, ϕ_0)

    mean_th_A_4 = th_mean_A(theory_time, α4, β4, γ4, ϕ_0, ψ_0)
    mean_th_B_4 = th_mean_B(theory_time, α4, β4, γ4, ϕ_0, ψ_0)
    var_th_A_4 = th_var_A(theory_time, α4, β4, ϕ_0)
    var_th_B_4 = th_var_B(theory_time, α4, β4, γ4, ψ_0, ϕ_0)
    cross_covar_th_4 = th_cross_covar(theory_time, β4, γ4, ϕ_0)

    mean_th_A_5 = th_mean_A(theory_time, α5, β5, γ5, ϕ_0, ψ_0)
    mean_th_B_5 = th_mean_B(theory_time, α5, β5, γ5, ϕ_0, ψ_0)
    var_th_A_5 = th_var_A(theory_time, α5, β5, ϕ_0)
    var_th_B_5 = th_var_B(theory_time, α5, β5, γ5, ψ_0, ϕ_0)
    cross_covar_th_5 = th_cross_covar(theory_time, β5, γ5, ϕ_0)

    mean_th_A_6 = th_mean_A(theory_time, α6, β6, γ6, ϕ_0, ψ_0)
    mean_th_B_6 = th_mean_B(theory_time, α6, β6, γ6, ϕ_0, ψ_0)
    var_th_A_6 = th_var_A(theory_time, α6, β6, ϕ_0)
    var_th_B_6 = th_var_B(theory_time, α6, β6, γ6, ψ_0, ϕ_0)
    cross_covar_th_6 = th_cross_covar(theory_time, β6, γ6, ϕ_0)


    mean_th_A_7 = th_mean_A(theory_time, α7, β7, γ7, ϕ_0, ψ_0)
    mean_th_B_7 = th_mean_B(theory_time, α7, β7, γ7, ϕ_0, ψ_0)
    var_th_A_7 = th_var_A(theory_time, α7, β7, ϕ_0)
    var_th_B_7 = th_var_B(theory_time, α7, β7, γ7, ψ_0, ϕ_0)
    cross_covar_th_7 = th_cross_covar(theory_time, β7, γ7, ϕ_0)


    mean_th_A_8 = th_mean_A(theory_time, α8, β8, γ8, ϕ_0, ψ_0)
    mean_th_B_8 = th_mean_B(theory_time, α8, β8, γ8, ϕ_0, ψ_0)
    var_th_A_8 = th_var_A(theory_time, α8, β8, ϕ_0)
    var_th_B_8 = th_var_B(theory_time, α8, β8, γ8, ψ_0, ϕ_0)
    cross_covar_th_8 = th_cross_covar(theory_time, β8, γ8, ϕ_0)
end

# Fano factor, correlation coefficient
begin
    ρs = [
        cross_covar_th_1 ./ (sqrt.(var_th_A_1 .* var_th_B_1)),
        cross_covar_th_2 ./ (sqrt.(var_th_A_2 .* var_th_B_2)),
        cross_covar_th_5 ./ (sqrt.(var_th_A_5 .* var_th_B_5)),
        cross_covar_th_6 ./ (sqrt.(var_th_A_6 .* var_th_B_6)),
        cross_covar_th_7 ./ (sqrt.(var_th_A_7 .* var_th_B_7)),
        cross_covar_th_8 ./ (sqrt.(var_th_A_8 .* var_th_B_8)),
    ]

    fanos_A = [
        var_th_A_1 ./ mean_th_A_1,
        var_th_A_2 ./ mean_th_A_2,
        var_th_A_5 ./ mean_th_A_5,
        var_th_A_6 ./ mean_th_A_6,
        var_th_A_7 ./ mean_th_A_7,
        var_th_A_8 ./ mean_th_A_8,
        
    ]

    th_fanos_A = [
        var_th_A_1 ./ mean_th_A_1,
        var_th_A_2 ./ mean_th_A_2,
        var_th_A_5 ./ mean_th_A_5,
        var_th_A_6 ./ mean_th_A_6,
        var_th_A_7 ./ mean_th_A_7,
        var_th_A_8 ./ mean_th_A_8,
        
    ]

    th_fanos_B = [
        var_th_B_1 ./ mean_th_B_1,
        var_th_B_2 ./ mean_th_B_2,
        var_th_B_5 ./ mean_th_B_5,
        var_th_B_6 ./ mean_th_B_6,
        var_th_B_7 ./ mean_th_B_7,
        var_th_B_8 ./ mean_th_B_8,
        
    ]
end


begin
    sm_ρ = [sim[2] for sim in sims_second_moment_data]
    sm_fano_A = [vars[i][1] ./ means[i][1] for i in eachindex(vars)]
    sm_fano_B = [vars[i][2] ./ means[i][2] for i in eachindex(vars)]
end

using CairoMakie
# ==================================== O(V^{0}) ==================================================================== #
using LaTeXStrings

begin
    sim_time = collect(sim_1["ts2"])
end
begin
    sim_time2 = collect(sim_1["ts"])
    sm_fano_A[1]
end
begin
    fig = Figure(fonts=(; sci="CMU Serif"), size = (700, 500), textcolor=fontcolor)

    plot1 = fig[1, 1]

    b_x_label = L"$t/\tau$"
    b_y_label = L"\rho(t)"

    a_x_label = L"$t/\tau$"
    a_y_label = L"F(t)"

    ax = CairoMakie.Axis(plot1,
        xlabel=b_x_label, ylabel=b_y_label, xlabelsize=30, ylabelsize=30, xticklabelsize=20, yticklabelsize=20, title=L"\text{ }",
        xticklabelfont=:sci, yticklabelfont=:sci, titlesize=27, aspect = 1, xtickcolor= fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
        bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, titlecolor=fontcolor, xlabelcolor =fontcolor,
        ylabelcolor =fontcolor, xgridvisible = false, ygridvisible = false, ytickformat = values -> [(LaTeXString(string(round(value, digits=2)))) for value in values], xtickformat = values -> [(LaTeXString(string((value)))) for value in values],)


    th1 = CairoMakie.lines!(ax, collect(theory_time) .* (2 * β), ρs[1],  linewidth=0.7, color=prussian_blue)
    th2 = CairoMakie.lines!(ax, collect(theory_time) .* (2 * β2), ρs[2],  linewidth=0.7, color=vista)
    th3 = CairoMakie.lines!(ax, collect(theory_time) .* (2 * β5), ρs[3],  linewidth=0.3, color=delft)
    th4 = CairoMakie.lines!(ax, collect(theory_time) .* (2 * β6), ρs[4],  linewidth=0.3, color=citron)
    #th5 = CairoMakie.lines!(ax, collect(theory_time), ρs[5],  linewidth=0.3, color=olivine)
    #th6 = CairoMakie.lines!(ax, collect(theory_time) , ρs[6],  linewidth=0.3, color=bittersweet)

    sm1 = CairoMakie.scatter!(ax, collect(sim_time) .* (2 * β), sm_ρ[1],  markersize=3, strokewidth=0.5, strokecolor = prussian_blue, color=:transparent)
    sm2 = CairoMakie.scatter!(ax, collect(sim_time) .* (2 * β2), sm_ρ[2],  markersize=3, strokewidth=0.5, strokecolor = vista, color=:transparent)
    sm3 = CairoMakie.scatter!(ax, collect(sim_time) .* (2 * β5), sm_ρ[5],  markersize=3, strokewidth=0.5, strokecolor = delft, color=:transparent)
    sm4 = CairoMakie.scatter!(ax, collect(sim_time) .* (2 * β6), sm_ρ[6],  markersize=3, strokewidth=0.5, strokecolor = citron, color=:transparent)
    #sm5 = CairoMakie.scatter!(ax, collect(sim_time), sm_ρ[7],  markersize=1, strokewidth=0.5, strokecolor = olivine, color=:transparent)
    #sm6 = CairoMakie.scatter!(ax, collect(sim_time) , sm_ρ[8],  markersize=5, strokewidth=1.3, strokecolor = bittersweet, color=:transparent)
    xlims!(ax, 0.0, 20.0)
    save("expo_rhos.png", fig)
end


#=


    ax2 = CairoMakie.Axis(plot2,
        xlabel=a_x_label, ylabel=a_y_label, xlabelsize=30, ylabelsize=30, xticklabelsize=20, yticklabelsize=20, title=L"\text{ }",
        xticklabelfont=:sci, yticklabelfont=:sci, titlesize=27, aspect = 1, xtickcolor= fontcolor, xticklabelcolor = fontcolor, ytickcolor=fontcolor, yticklabelcolor = fontcolor, 
        bottomspinecolor=fontcolor, leftspinecolor=fontcolor, rightspinecolor=fontcolor,topspinecolor=fontcolor, titlecolor=fontcolor, xlabelcolor =fontcolor, xscale=log10,
        ylabelcolor =fontcolor, xgridvisible = false, ygridvisible = false, ytickformat = values ->[(LaTeXString(string(round(value, digits=2)))) for value in values])

        f_A_th_1 = CairoMakie.lines!(ax2, collect(theory_time) .* ((0.5 * β)), th_fanos_A[1],  linewidth=0.7, color=wine)
        f_B_th_1 = CairoMakie.lines!(ax2, collect(theory_time) .* ((β + γ)), th_fanos_B[1],  linewidth=0.7, color=blush)
        f_A_sm_1 = CairoMakie.scatter!(ax2, collect(sim_time2) .* ((0.5 * β)), sm_fano_A[1],  markersize=3, strokewidth=0.5, strokecolor = wine, color=:transparent)
        f_B_sm_1 = CairoMakie.scatter!(ax2, collect(sim_time2) .* ((β + γ)), sm_fano_B[1],  markersize=3, strokewidth=0.5, strokecolor = blush, color=:transparent)


        f_A_th_2 = CairoMakie.lines!(ax2, collect(theory_time) .* ((0.5 * β2)), th_fanos_A[2],  linewidth=0.7, color=citron)
        f_B_th_2 = CairoMakie.lines!(ax2, collect(theory_time) .* ((β2 + γ2)), th_fanos_B[2],  linewidth=0.7, color=delft)
        f_A_sm_2 = CairoMakie.scatter!(ax2, collect(sim_time2) .* ((0.5 * β2)), sm_fano_A[2],  markersize=3, strokewidth=0.5, strokecolor = citron, color=:transparent)
        f_B_sm_2 = CairoMakie.scatter!(ax2, collect(sim_time2) .* ((β2 + γ2)), sm_fano_B[2],  markersize=3, strokewidth=0.5, strokecolor = delft, color=:transparent)

        f_A_th_3 = CairoMakie.lines!(ax2, collect(theory_time) .* ((0.5 * β5)), th_fanos_A[3],  linewidth=0.7, color=tea)
        f_B_th_3 = CairoMakie.lines!(ax2, collect(theory_time) .* ((β5 + γ5)), th_fanos_B[3],  linewidth=0.7, color=dark_green)
        f_A_sm_3 = CairoMakie.scatter!(ax2, collect(sim_time2) .* ((0.5 * β5)), sm_fano_A[5],  markersize=3, strokewidth=0.5, strokecolor = tea, color=:transparent)
        f_B_sm_3 = CairoMakie.scatter!(ax2, collect(sim_time2) .* ((β5 + γ5)), sm_fano_B[5],  markersize=3, strokewidth=0.5, strokecolor = dark_green, color=:transparent)

        f_A_th_4 = CairoMakie.lines!(ax2, collect(theory_time) .* ((0.5 * β6)), th_fanos_A[4],  linewidth=0.7, color=prussian_blue)
        f_B_th_4 = CairoMakie.lines!(ax2, collect(theory_time) .* ((β6 + γ6)), th_fanos_B[4],  linewidth=0.7, color=vista)
        f_A_sm_4 = CairoMakie.scatter!(ax2, collect(sim_time2) .* ((0.5 * β6)), sm_fano_A[6],  markersize=3, strokewidth=0.5, strokecolor = prussian_blue, color=:transparent)
        f_B_sm_4 = CairoMakie.scatter!(ax2, collect(sim_time2) .* ((β6 + γ6)), sm_fano_B[6],  markersize=3, strokewidth=0.5, strokecolor = vista, color=:transparent)
        xlims!(ax2, 1.0, 50)
=#