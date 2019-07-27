using ApproxFun, WienerHopf, Test
import  WienerHopf: fourtoone, ifourtoone


@testset "4-to-1" begin
    @test ifourtoone(fourtoone(0.1)) ≈ ifourtoone(1,fourtoone(0.1)) ≈ 0.1
    @test fourtoone(ifourtoone(1,0.1+0.1im)) ≈ fourtoone(ifourtoone(2,0.1+0.1im)) ≈ 
        fourtoone(ifourtoone(3,0.1+0.1im)) ≈ fourtoone(ifourtoone(4,0.1+0.1im)) ≈ 0.1+0.1im

    f = Fun(x -> 1/sqrt(x-im) + 1/(x-im), SqrtLine())
    @test ncoefficients(f) ≤ 110 # spectral convergence
    @test f(0.1) ≈ 1/sqrt(0.1-im) + 1/(0.1-im)
end