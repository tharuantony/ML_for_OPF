using Distributions

μ = 0.0
σ = 1.0
lower = 0.8
upper = 1.2

d = Distributions.truncated(Normal(μ, σ), lower, upper)

samples = rand(d, 5)

println(samples)
