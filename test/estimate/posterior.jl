using DSGE
using HDF5, Test
using Dates

path = dirname(@__FILE__)

custom_settings = Dict{Symbol, Setting}(
    :date_forecast_start  => Setting(:date_forecast_start, quartertodate("2015-Q4")))
m = AnSchorfheide(custom_settings = custom_settings, testing = true)

file = "$path/../reference/posterior.h5"
data = Matrix(h5read(file, "data")')
lh_expected = h5read(file, "likelihood")
post_expected = h5read(file, "posterior")

@testset "Check likelihood and posterior calculations" begin
    lh = likelihood(m, data)
    @test lh_expected ≈ lh

    post = posterior(m, data)
    @test post_expected ≈ post

    x = map(α->α.value, m.parameters)
    post_at_start = posterior!(m, x, data)
    @test post_expected ≈ post_at_start

    # Ensure if we are not evaluating at start vector, then we do not get the reference
    # posterior
    y = x .+ 0.01
    post_not_at_start = posterior!(m, y, data)
    ϵ = 1.0
    @test abs(post_at_start - post_not_at_start) > ϵ
end
