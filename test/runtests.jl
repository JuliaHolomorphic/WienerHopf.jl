using WienerHopf, ApproxFun, RiemannHilbert, SingularIntegralEquations, Test
import ApproxFun: mappoint, setcanonicaldomain
import RiemannHilbert: orientedleftendpoint, fpcauchymatrix, fpstieltjesmatrix, collocationpoints, evaluationmatrix!, undirected, evaluationmatrix
import WienerHopf: fourtoone, ifourtoone, sqrtline_tocanonical


@testset "SqrtLine" begin
    @test ifourtoone(0.001-eps()) ≈ ifourtoone(0.001+eps()) # taylor v explicit
    @test ifourtoone(-0.001-eps()) ≈ ifourtoone(-0.001+eps()) # taylor v explicit
    @test ifourtoone(-0.001im-eps()im) ≈ ifourtoone(-0.001im+eps()im) # taylor v explicit
    @test sqrtline_tocanonical(-eps()) ≈ sqrtline_tocanonical(eps()) atol=1E-15
    @test sqrtline_tocanonical(-eps()) ≈ sqrtline_tocanonical(0) atol=1E-15
    @test 0 ≈ sqrtline_tocanonical(0) atol=1E-15

    @test ifourtoone(fourtoone(0.1)) ≈ ifourtoone(1,fourtoone(0.1)) ≈ 0.1
    @test ifourtoone(fourtoone(-0.1)) ≈ ifourtoone(1,fourtoone(-0.1)) ≈ -0.1
    @test fourtoone(ifourtoone(1,0.1+0.1im)) ≈ fourtoone(ifourtoone(2,0.1+0.1im)) ≈ 
        fourtoone(ifourtoone(3,0.1+0.1im)) ≈ fourtoone(ifourtoone(4,0.1+0.1im)) ≈ 0.1+0.1im
        @test fourtoone(ifourtoone(1,-0.1-0.1im)) ≈ fourtoone(ifourtoone(2,-0.1-0.1im)) ≈ 
        fourtoone(ifourtoone(3,-0.1-0.1im)) ≈ fourtoone(ifourtoone(4,-0.1-0.1im)) ≈ -0.1-0.1im        

    @test angle(fromcanonical(SqrtLine{1/4}(), 0.1)) ≈ 0.7853981633974483
    @test angle(fromcanonical(SqrtLine{1/4}(), -0.1)) ≈ 0.7853981633974483-π
    @test isreal(tocanonical(SqrtLine{1/4}(), 0.1exp(im*π/4)))
    @test isreal(tocanonical(SqrtLine{1/4}(), -0.1exp(im*π/4)))

    f = Fun(x -> 1/sqrt(x-im) + 1/(x-im) + 1/sqrt(x+im-1) + 1/(x+im-1), SqrtLine())
    @test ncoefficients(f) ≤ 250 # spectral convergence
    @test f(0.1) ≈ 1/sqrt(0.1-im) + 1/(0.1-im) + 1/sqrt(0.1+im-1) + 1/(0.1+im-1)
end

@testset "SqrtLine Cauchy" begin
    f = Fun(x -> 1/sqrt(x-im) + 1/(x-im) + 1/sqrt(x+im-1) + 1/(x+im-1), SqrtLine())
    @test cauchy(f, 1+im) ≈ 1/sqrt((1+im)+im-1) + 1/((1+im)+im-1) 
    @test cauchy(f, -1+im) ≈ 1/sqrt((-1+im)+im-1) + 1/((-1+im)+im-1) 
    @test cauchy(f, 1-im) ≈ -1/sqrt((1-im)-im) - 1/((1-im)-im) 
    @test cauchy(f, -1-im) ≈ -1/sqrt((-1-im)-im) - 1/((-1-im)-im) 

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

    S = Legendre(SqrtLine{-1/4}())
    f = Fun(α -> -k*sin(θ₀)/(γ(α)*(α-k*cos(θ₀))), S)
    C = fpcauchymatrix(S, ncoefficients(f), ncoefficients(f))
    @test norm(C) ≤ 1500
    pts = collocationpoints(space(f), ncoefficients(f))
    @test (C*f.coefficients)[50] ≈ cauchy(f,pts[50]-10eps()im) ≈ cauchy(f,pts[50]⁻)
    @test (C*f.coefficients)[100] ≈ cauchy(f,pts[100]-10eps()im) ≈ cauchy(f,pts[100]⁻)
    @test (C*f.coefficients)[229] ≈ cauchy(f,pts[229]-10eps()im) ≈ cauchy(f,pts[229]⁻)
end

@testset "RHP" begin
    S = Legendre(SqrtLine{-1/4}())
    n = 256
    pts = collocationpoints(S, n)
    Cm = fpcauchymatrix(S,n,n)[2:end-1,:]
    E = evaluationmatrix(S,pts[2:end-1],n)
    Cp = E + Cm
    k = 1
    θ₀ = 0.1
    γ = α -> isinf(α) ? complex(Inf) : im*sqrt(k-α)*sqrt(α+k)
    f = α -> isinf(α) ? zero(α) : -k*sin(θ₀)/(γ(α)*(α-k*cos(θ₀)))

    Hp = α -> k*sin(θ₀)/(α-k*cos(θ₀)) * (1/sqrt(α+k) - 1/sqrt(k+k*cos(θ₀)))
    Hm = α -> k*sin(θ₀) / (sqrt(k+k*cos(θ₀))*(α-k*cos(θ₀)))

    Φdp = α -> -sqrt(α+k)Hp(α)
    Dm = α -> -Hm(α)/(im*sqrt(k-α))

    @test Φdp(0.0) + γ(0.0)Dm(0.0) ≈ -k*sin(θ₀)/(0.0-k*cos(θ₀))

    v = Fun(α -> Φdp(α*exp(-im*π/4))-Dm(α*exp(-im*π/4)), SqrtLine())
    @which evaluate(v.coefficients,space(v),0.0)
    @which tocanonical(space(v),0.000001)
    @test v(0.0) ≈ Φdp(0.0)-Dm(0.0)
    @test cauchy(v,0.1+10eps()im)-cauchy(v,0.1-10eps()im) ≈ v(0.1)
    @test cauchy(v,-0.1+10eps()im)-cauchy(v,-0.1-10eps()im) ≈ v(-0.1)
    @test cauchy(v,100eps()im)-cauchy(v,-100eps()im) ≈ v(0.0)
    

    v = Fun(α -> Φdp(α)-Dm(α), S)
    @test cauchy(v,2.0) ≈ Φdp(2.0)
    @test cauchy(v,-2.0) ≈ Dm(-2.0)

    @test Φdp(im) ≈ cauchy(v,im)
    h = 0.00001; @test Φdp(h) ≈ cauchy(v,h)
    h = 0.00001im; @test Φdp(h) ≈ cauchy(v,h)

    @test cauchy(v,pts[50]+100eps()im)-cauchy(v,pts[50]-100eps()im) ≈ v(pts[50])

    @test cauchy(v,0.0+10eps()im) + γ(0.0)*Dm(0.0) ≈ Φdp(0.0) + γ(0.0)*Dm(0.0) ≈ -k*sin(θ₀)/(0.0-k*cos(θ₀))
    @test (Cm*v.coefficients)[50] ≈ cauchy(v,pts[50]-100eps()im)
    @test (Cp*v.coefficients)[50] ≈ cauchy(v,pts[50]+100eps()im)

    L = Diagonal(inv.(γ.(pts[2:end-1])))*Cp .+ Cm 
    norm(L*v.coefficients .- f.(pts[2:end-1]))
    u = [fill(1.0,1,n); L; ((-1.0).^(0:n-1))'] \ f.(pts)
    @test u ≈ v.coefficients
end
