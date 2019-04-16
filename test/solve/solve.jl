using HDF5
using Test
using DSGE

path = dirname(@__FILE__)

myfile = "$path/../reference/solve.h5"
TTT_expected = h5read(myfile, "TTT")
CCC_expected = h5read(myfile, "CCC")
RRR_expected = h5read(myfile, "RRR")

m = AnSchorfheide()
TTT, RRR, CCC = solve(m)

@testset "Check state-space system matches reference" begin
    @test @test_matrix_approx_eq TTT_expected TTT
    @test @test_matrix_approx_eq RRR_expected RRR
    @test @test_matrix_approx_eq CCC_expected CCC
end
