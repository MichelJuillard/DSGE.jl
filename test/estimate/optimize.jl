using DSGE
using HDF5
using Test
path = dirname(@__FILE__)

# Test in model optimization
custom_settings = Dict{Symbol, Setting}(
    :date_forecast_start  => Setting(:date_forecast_start, quartertodate("2015-Q4")))
m = AnSchorfheide(custom_settings = custom_settings, testing = true)


file = "$path/../reference/optimize.h5"
x0 = h5read(file, "params")
data = Matrix(h5read(file, "data")')
minimizer = h5read(file, "minimizer")
minimum = h5read(file, "minimum")
H_expected = h5read(file, "H")

# See src/estimate/estimate.jl
update!(m, x0)
n_iterations = 3

x0 = Float64[p.value for p in m.parameters]

out, H = optimize!(m, data; iterations=n_iterations)

@testset "Check optimize minimizers are the same" begin
    @test @test_matrix_approx_eq minimizer out.minimizer
    @test minimum ≈ out.minimum atol=5e-7
    @test @test_matrix_approx_eq H_expected H
end

nothing
