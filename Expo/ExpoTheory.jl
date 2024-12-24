module ExpoTheory

export th_cross_covar, th_var_A, th_var_B, th_mean_A, th_mean_B


function th_mean_A(ts, α, β, γ, ϕ_0, ψ_0)
    return  @. (ϕ_0 * exp(-β *ts) - (α/β)  * exp(- β * ts) + (α/β))
end

function th_mean_B(ts, α, β, γ, ϕ_0, ψ_0)
    theory_b = ((α * exp.(- β .* ts) .- α .- (β * ϕ_0) .* exp.(-β .* ts))/(β - γ))
    theory_b .+= (α * β) / (γ * (β - γ))
    theory_b .+= ((ψ_0 + ((β * ϕ_0)/(β - γ)) - ((α * β)/(γ * (β - γ)))) .* exp.(- γ .* ts))
    
    return theory_b
end
function th_cross_covar(ts, β, γ, ϕ0)
    # Vectorize the computation over the time array ts
    return @. -(
        (β * ϕ0 * exp(-2β * ts) * (exp(ts * (β - γ)) - 1)) / (β - γ)
    )
end


function th_var_B(t, α, β, γ, ψ0, ϕ0)
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

function th_var_A(t, α, β, ϕ0)
    # Using the @. macro for vectorized operations
    exp_beta_t = @. exp(β * t)
    exp_neg2beta_t = @. exp(-2 * β * t)
    
    @. exp_neg2beta_t * (exp_beta_t - 1) * (β * ϕ0 + α * exp_beta_t) / β
end

end;