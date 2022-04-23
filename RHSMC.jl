function ψ(x0,v0,n;ϵ,U0,U,λ)
    D = length(x0)
    xvec = zeros(n+1,D)
    vvec = zeros(n+1,D)
    xvec[1,:] = x0
    vvec[1,:] = v0
    U0VEC = zeros(n+1)
    UVEC  = zeros(n+1)
    U0VEC[1],grad0 = U0(x0,grad=true)
    UVEC[1],grad1  = U(x0,grad=true)
    for i = 1:n
        tempv = vvec[i,:] .- ϵ/2*((1-λ)*grad0+λ*grad1)
        xvec[i+1,:] = xvec[i,:] .+ ϵ*tempv
        U0VEC[i+1],grad0 = U0(xvec[i+1,:],grad=true)
        UVEC[i+1],grad1  = U(xvec[i+1,:],grad=true)
        vvec[i+1,:] = tempv .- ϵ/2*((1-λ)*grad0+λ*grad1)
    end
    return xvec,vvec,U0VEC,UVEC
end

function RecycledHSMC(N::Int,M::Int,model,)
