N = 10000
M = 100
D = 61
α = 0.5
ϵ = 0.2

UVec = Array{Any,1}(undef,0); push!(UVec,x->U0(x))
gradUVec = Array{Any,1}(undef,0);push!(gradUVec,x->gradient(UVec[1],x))
HVec = Array{Any,1}(undef,0);push!(HVec,(x,v)->UVec[1](x)+1/2*norm(v)^2)

X = Array{Matrix,1}(undef,0);push!(X,zeros(N,D))
V = Array{Matrix,1}(undef,0);push!(V,zeros(N,D))
λ = zeros(1)
logW = zeros(N,1)
W = zeros(N,1)
P = div(N,M)
for i = 1:N
    X[1][i,:] = rand(initDist)
    V[1][i,:] = randn(D)
    logW[i,1] = -HVec[1](X[1][i,:],V[1][i,:])
end
MAX = findmax(logW[:,1])[1]
W[:,1] = exp.(logW[:,1] .- MAX)/sum(exp.(logW[:,1] .- MAX))
t = 1

t +=1 # move to the next step
push!(X,zeros(N,D));push!(V,zeros(N,D))
A = vcat(fill.(1:N,rand(Multinomial(M,W[:,t-1])))...)# resample "starting points"
targetU(x) = (1-λ[t-1])*U0(x) + λ[t-1]*U(x)
targetgradU(x) = gradient(targetU,x)
targetH(x,v) = targetU(x)+1/2*norm(v)^2
for n = 1:M
    x0 = X[t-1][A[n],:]
    v0 = randn(D)
    newx,newv = MH(x0,v0,P,ϵ,targetgradU,targetH)
    for i = 1:P
        X[t][(n-1)*P+i,:] = newx[i,:]
        V[t][(n-1)*P+i,:] = newv[i,:]
    end
end
tar(λ) = ESS(X[t],V[t],λ,targetH,U0,U) - α*N
newλ = find_zero(tar,(λ[end-1],10.0),Bisection())
if newλ >1.0
    push!(λ,1.0)
else
    push!(λ,newλ)
end
newH(x,v) = (1-newλ)*U0(x) + newλ*U(x) + 1/2*norm(v)^2

logW = hcat(logW,zeros(N))
W = hcat(W,zeros(N))
logW[:,t] = getW(X[t],V[t],newH,targetH)
MAX = findmax(logW[:,t])[1]
W[:,t] = exp.(logW[:,t] .- MAX)/sum(exp.(logW[:,t] .- MAX))  
1/sum(W[:,end].^2)


function brent(f::Function, x0::Number, x1::Number, args::Tuple=();xtol::AbstractFloat=1e-7, ytol=2eps(Float64),maxiter::Integer=50)
    EPS = eps(Float64)
    y0 = f(x0,args...)
    y1 = f(x1,args...)
    if abs(y0) < abs(y1)
        # Swap lower and upper bounds.
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    end
    x2 = x0
    y2 = y0
    x3 = x2
    bisection = true
    for _ in 1:maxiter
        # x-tolerance.
        if abs(x1-x0) < xtol
            return x1
        end

        # Use inverse quadratic interpolation if f(x0)!=f(x1)!=f(x2)
        # and linear interpolation (secant method) otherwise.
        if abs(y0-y2) > ytol && abs(y1-y2) > ytol
            x = x0*y1*y2/((y0-y1)*(y0-y2)) + x1*y0*y2/((y1-y0)*(y1-y2)) + x2*y0*y1/((y2-y0)*(y2-y1))
        else
            x = x1 - y1 * (x1-x0)/(y1-y0)
        end

        # Use bisection method if satisfies the conditions.
        delta = abs(2EPS*abs(x1))
        min1 = abs(x-x1)
        min2 = abs(x1-x2)
        min3 = abs(x2-x3)
        if (x < (3x0+x1)/4 && x > x1) || (bisection && min1 >= min2/2) || (!bisection && min1 >= min3/2) || (bisection && min2 < delta) || (!bisection && min3 < delta)
            x = (x0+x1)/2
            bisection = true
        else
            bisection = false
        end

        y = f(x,args...)
        # y-tolerance.
        if abs(y) < ytol
            return x
        end
        x3 = x2
        x2 = x1
        if sign(y0) != sign(y)
            x1 = x
            y1 = y
        else
            x0 = x
            y0 = y
        end
        if abs(y0) < abs(y1)
        # Swap lower and upper bounds.
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        end
    end
    error("Max iteration exceeded")
end

@time brent(tar,λ[end-1],10.0)

@time newλ = find_zero(tar,(λ[end-1],1.65*λ[end-1]),Bisection())

@time brent(tar,λ[end-1],2.0*λ[end-1])

function bisect(f,a,b)
    midpoint = (a+b)/2
    left = f(a)
    right = f(b)
    while abs(f(midpoint)) > 1e-7
        new = f(midpoint)
        println(new)
        if new*left > 0
            a = midpoint
            left = new
        else
            b = midpoint
            right = new
        end
        midpoint = (a+b)/2
    end
    return midpoint
end

@time bisect(tar,λ[end-1],2*λ[end-1])