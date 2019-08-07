module WienerHopf
using ApproxFun, SingularIntegralEquations, RiemannHilbert, ForwardDiff

import ApproxFunBase: SegmentDomain, fromcanonical, tocanonical, angle,
                        reverseorientation, isambiguous, arclength,
                        leftendpoint, rightendpoint, complexlength,
                        AnyDomain, tocanonicalD, fromcanonicalD,
                        setcanonicaldomain, mappoint
import SingularIntegralEquations: stieltjes, stieltjesmoment!, undirected
import Base: convert, isempty, +, -, *, ==    
import ForwardDiff: derivative
import RiemannHilbert: RiemannDual, finitepart, 
            fpleftstieltjesmoment!, fprightstieltjesmoment!,
            collocationpoints, fpcauchymatrix, evaluationmatrix
            

export SqrtLine, collocationpoints, fpcauchymatrix, evaluationmatrix

"""
    SqrtLine{a}(c)

represents the line at angle `a` in the complex plane, centred at `c`.
This is encoded by a map from -1..1 that allows rapid expansion of 
functions with sqrt-decay.
"""
struct SqrtLine{angle,T<:Number} <: SegmentDomain{T}
    center::T

    #TODO get this inner constructor working again.
    SqrtLine{angle,T}(c) where {angle,T} = new{angle,T}(c)
    SqrtLine{angle,T}() where {angle,T} = new{angle,T}(zero(T))
end

SqrtLine{a}(c::Number) where {a} = SqrtLine{a,typeof(c)}(c)
SqrtLine{a}() where {a} = SqrtLine{a,Float64}()
SqrtLine() = SqrtLine{false}()

angle(d::SqrtLine{a}) where {a} = a*π

reverseorientation(d::SqrtLine{true}) = SqrtLine{false}(d.center)
reverseorientation(d::SqrtLine{false}) = SqrtLine{true}(d.center)
reverseorientation(d::SqrtLine{a}) where {a} = SqrtLine{a-1}(d.center)




isambiguous(d::SqrtLine) = isnan(d.center)
convert(::Type{Domain{T}},d::SqrtLine{a}) where {a,T<:Number} = SqrtLine{a,T}(d.center)
convert(::Type{SqrtLine{a,T}},::AnyDomain) where {a,T<:Number} = SqrtLine{a,T}(NaN)
convert(::Type{IT},::AnyDomain) where {IT<:SqrtLine}=SqrtLine(NaN,NaN)

## Map interval


##TODO non-1 alpha,beta

isempty(::SqrtLine) = false

fourtoone(x) = (x+x^3)/(1-x^2)^2
ifourtoone(α) = if abs(α) ≤ 0.001
        α - 3α^3 + 22α^5 # Taylor series
    else
        (1 + sqrt(1 + 16α^2) - sqrt(2)sqrt(1 + sqrt(1 + 16α^2)))/(4α)
    end

function ifourtoone(k, α) 
    x = ifourtoone(α)
    k == 1 && return x
    k == 2 && return inv(x)
    c = x + inv(x) - inv(α)
    k == 3 && return (-c+sqrt(c^2-4))/2
    k == 4 && return (-c-sqrt(c^2-4))/2
    throw(ArgumentError("$k"))
end

sqrtline_tocanonical(x) = ifourtoone(x)
sqrtline_fromcanonical(x) = fourtoone(x)

sqrtline_tocanonicalD(x) = derivative(ifourtoone,x)
sqrtline_fromcanonicalD(x) = derivative(fourtoone,x)


tocanonical(d::SqrtLine,x) = sqrtline_tocanonical(cis(-angle(d)).*(x-d.center))
tocanonical(d::SqrtLine{false},x) = sqrtline_tocanonical(x-d.center)
tocanonical(d::SqrtLine{true},x) = sqrtline_tocanonical(d.center-x)

tocanonicalD(d::SqrtLine,x) = cis(-angle(d)).*sqrtline_tocanonicalD(cis(-angle(d)).*(x-d.center))
tocanonicalD(d::SqrtLine{false},x) = sqrtline_tocanonicalD(x-d.center)
tocanonicalD(d::SqrtLine{true},x) = -sqrtline_tocanonicalD(d.center-x)

fromcanonical(d::SqrtLine,x) = cis(angle(d))*sqrtline_fromcanonical(x)+d.center
fromcanonical(d::SqrtLine{false},x) = sqrtline_fromcanonical(x)+d.center
fromcanonical(d::SqrtLine{true},x) = -sqrtline_fromcanonical(x)+d.center

fromcanonicalD(d::SqrtLine,x) = cis(angle(d))*sqrtline_fromcanonicalD(x)
fromcanonicalD(d::SqrtLine{false},x) = sqrtline_fromcanonicalD(x)
fromcanonicalD(d::SqrtLine{true},x) = -sqrtline_fromcanonicalD(x)

invfromcanonicalD(d::SqrtLine,x) = cis(-angle(d))*sqrtline_invfromcanonicalD(x)
invfromcanonicalD(d::SqrtLine{false},x) = sqrtline_invfromcanonicalD(x)
invfromcanonicalD(d::SqrtLine{true},x) = -sqrtline_invfromcanonicalD(x)

==(d::SqrtLine{a},m::SqrtLine{a}) where {a} = d.center == m.center

# algebra
*(c::Real,d::SqrtLine{false}) = SqrtLine{sign(c)>0 ? false : true}(isapprox(d.center,0) ? d.center : c*d.center,d.α,d.β)
*(c::Real,d::SqrtLine{true}) = SqrtLine{sign(c)>0 ? true : false}(isapprox(d.center,0) ? d.center : c*d.center,d.α,d.β)
*(c::Number,d::SqrtLine) = SqrtLine(isapprox(d.center,0) ? d.center : c*d.center,angle(d)+angle(c),d.α,d.β)
*(d::SqrtLine,c::Number) = c*d
for OP in (:+,:-)
    @eval begin
        $OP(c::Number,d::SqrtLine{a}) where {a} = SqrtLine{a}($OP(c,d.center),d.α,d.β)
        $OP(d::SqrtLine{a},c::Number) where {a} = SqrtLine{a}($OP(d.center,c),d.α,d.β)
    end
end

# algebra
arclength(d::SqrtLine) = Inf
leftendpoint(d::SqrtLine) = -Inf
rightendpoint(d::SqrtLine) = Inf
complexlength(d::SqrtLine) =Inf


# Sum over all inverses of fromcanonical, see [Olver,2014]

fpstieltjes(s,f,z) = finitepart(stieltjes(s,f,RiemannDual(z,1)))

function stieltjes(S::Space{<:SqrtLine},f,z)
    if domain(S) == SqrtLine()
        # TODO: rename tocanonical
        s = setcanonicaldomain(S)
        u = undirected(z)
        stieltjes(s,f,ifourtoone(1,z))+stieltjes(s,f,ifourtoone(2,u))+
            stieltjes(s,f,ifourtoone(3,u))+stieltjes(s,f,ifourtoone(4,u))-
            2fpstieltjes(s,f,1)-2fpstieltjes(s,f,-1)
    else
        stieltjes(setdomain(S,SqrtLine()),f,mappoint(domain(S),SqrtLine(),z))
    end
end

function stieltjesmoment!(ret,S::PolynomialSpace{<:SqrtLine},z,filter=identity)
    if domain(S) == SqrtLine()
        s = setcanonicaldomain(S)
        u = complex(undirected(z))
        tmp = similar(ret)
        stieltjesmoment!(ret, s, ifourtoone(1,z),filter)
        stieltjesmoment!(tmp, s, ifourtoone(2,u),filter); ret .+= tmp
        stieltjesmoment!(tmp, s, ifourtoone(3,u),filter); ret .+= tmp
        stieltjesmoment!(tmp, s, ifourtoone(4,u),filter); ret .+= tmp
        fpleftstieltjesmoment!(tmp, s); ret .-= 2 .* tmp
        fprightstieltjesmoment!(tmp, s); ret .-= 2 .* tmp
        ret
    else
        stieltjesmoment!(ret,setdomain(S,SqrtLine()),mappoint(domain(S),SqrtLine(),z),filter)
    end
end


fprightstieltjesmoment!(V, sp::PolynomialSpace{<:SqrtLine}) = fill!(V,0)
fpleftstieltjesmoment!(V, sp::PolynomialSpace{<:SqrtLine}) = fill!(V,0)

end # module
