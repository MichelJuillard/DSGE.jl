using Test
using DSGE
using HDF5

path = dirname(@__FILE__)

m = AnSchorfheide()
Γ0, Γ1, C, Ψ, Π = eqcond(m)
stake = 1 + 1e-6
G1, C, impact, fmat, fwt, ywt, gev, eu, loose = gensys(Γ0, Γ1, C, Ψ, Π, stake)

myfile = "$path/../reference/gensys.h5"
G1_exp = h5read(myfile, "G1_gensys")
C_exp = h5read(myfile,"C_gensys")
impact_exp = h5read(myfile, "impact")
eu_exp = h5read(myfile, "eu")

@testset "Check gensys outputs match reference" begin
    @test @test_matrix_approx_eq G1_exp G1
    @test @test_matrix_approx_eq C_exp C
    @test @test_matrix_approx_eq impact_exp impact

    @test isequal(eu_exp, eu)
end
