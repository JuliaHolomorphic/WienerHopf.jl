using WienerHopf, ApproxFun, RiemannHilbert, SingularIntegralEquations, Test
import RiemannHilbert: orientedleftendpoint, fpcauchymatrix, collocationpoints, evaluationmatrix!
import WienerHopf: fourtoone, ifourtoone


@testset "SqrtLine" begin
    @test ifourtoone(fourtoone(0.1)) ≈ ifourtoone(1,fourtoone(0.1)) ≈ 0.1
    @test fourtoone(ifourtoone(1,0.1+0.1im)) ≈ fourtoone(ifourtoone(2,0.1+0.1im)) ≈ 
        fourtoone(ifourtoone(3,0.1+0.1im)) ≈ fourtoone(ifourtoone(4,0.1+0.1im)) ≈ 0.1+0.1im

    @test angle(fromcanonical(SqrtLine{1/4}(), 0.1)) ≈ 0.7853981633974483
    @test isreal(tocanonical(SqrtLine{1/4}(), 0.1exp(im*π/4)))

    f = Fun(x -> 1/sqrt(x-im) + 1/(x-im) + 1/sqrt(x+im-1) + 1/(x+im-1), SqrtLine())
    @test ncoefficients(f) ≤ 250 # spectral convergence
    @test f(0.1) ≈ 1/sqrt(0.1-im) + 1/(0.1-im) + 1/sqrt(0.1+im-1) + 1/(0.1+im-1)
end

@testset "SqrtLine Cauchy" begin
    f = Fun(x -> 1/sqrt(x-im) + 1/(x-im) + 1/sqrt(x+im-1) + 1/(x+im-1), SqrtLine())
    @test cauchy(f, 1+im) ≈ 1/sqrt((1+im)+im-1) + 1/((1+im)+im-1) 
    @test cauchy(f, 1-im) ≈ -1/sqrt((1-im)-im) - 1/((1-im)-im) 

    f = Fun(x -> 1/sqrt(x-im) + 1/(x-im) + 1/sqrt(x+im-1) + 1/(x+im-1), Legendre(SqrtLine()))
    @test cauchy(f, 1+im) ≈ 1/sqrt((1+im)+im-1) + 1/((1+im)+im-1) 
    @test cauchy(f, 1-im) ≈ -1/sqrt((1-im)-im) - 1/((1-im)-im) 

    C = fpcauchymatrix(space(f), ncoefficients(f), ncoefficients(f))
    pts = collocationpoints(space(f), ncoefficients(f))
    @test (C*f.coefficients)[5] ≈ cauchy(f,pts[5]-eps()im)

    k = 1
    γ = α -> im*sqrt(k-α)*sqrt(α+k)
    θ₀ = 0.1
    f = Fun(α -> -k*sin(θ₀)/(γ(α)*(α-k*cos(θ₀))), SqrtLine{-1/4}())
    @test ncoefficients(f) ≤ 300
    @test f(exp(-im*π/4)) ≈ (α->(-k*sin(θ₀)/(γ(α)*(α-k*cos(θ₀)))))(exp(-im*π/4))
    @test cauchy(f,exp(-im*π/4)+10eps())-cauchy(f,exp(-im*π/4)-10eps()) ≈ f(exp(-im*π/4))

    f = Fun(α -> -k*sin(θ₀)/(γ(α)*(α-k*cos(θ₀))), Legendre(SqrtLine{-1/4}()))

    C = fpcauchymatrix(space(f), ncoefficients(f), ncoefficients(f))
    pts = collocationpoints(space(f), ncoefficients(f))
    @test (C*f.coefficients)[100] ≈ cauchy(f,pts[100]-eps()im)
end

@testset "RHP" begin
    S = Legendre(SqrtLine{-1/4}())
    n = 300
    pts = collocationpoints(S, n)
    Cm = fpcauchymatrix(S,n,n)
    Cp = I + Cm
    k = 1
    θ₀ = 0.1
    γ = α -> isinf(α) ? complex(Inf) : im*sqrt(k-α)*sqrt(α+k)
    f = α -> isinf(α) ? zero(α) : -k*sin(θ₀)/(γ(α)*(α-k*cos(θ₀)))
    L = Diagonal(inv.(γ.(pts)))*Cp .+ Cm 
    u = [fill(1.0,1,n); L[2:end-1,:]; ((-1.0).^(0:n-1))'] \ f.(pts)
end
