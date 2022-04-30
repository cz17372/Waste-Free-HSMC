module HMC
using Distributions,LinearAlgebra,ProgressMeter
function ψ(x0,v0,n;ϵ,U,keeptrej=false)
    xvec = zeros(n+1,length(x0))
    vvec = zeros(n+1,length(x0))
    xvec[1,:] = x0
    vvec[1,:] = v0
    U0VEC = zeros(n+1)
    UVEC  = zeros(n+1)
    _,grad  = U(x0,requires_grad=true)
    for i = 1:n
        tempv = vvec[i,:] .- ϵ/2*grad
        xvec[i+1,:] = xvec[i,:] .+ ϵ*tempv
        _,grad = U(xvec[i+1,:],requires_grad=true)
        vvec[i+1,:] = tempv .- ϵ/2*grad
    end
    if keeptrej
        return xvec,vvec
    else
        return xvec[end,:],vvec[end,:]
    end
end
function run(N,ϵ,L;U,x0,acc_prob=false)
    output = zeros(N+1,length(x0))
    output[1,:] = x0
    accept = 0
    @showprogress 1 for n = 2:(N+1)
        #velocity refreshment
        v0 = randn(length(x0))
        newx,newv = ψ(output[n-1,:],v0,L,ϵ=ϵ,U=U)
        oldH = U(output[n-1,:],requires_grad=false) + 1/2*norm(v0)^2
        newH = U(newx,requires_grad=false) + 1/2*norm(newv)^2
        if log(rand()) < -newH+oldH
            output[n,:] = newx
            accept += 1
        else
            output[n,:] = output[n-1,:]
        end
    end
    if acc_prob
        return output,accept/N
    else
        return output
    end
end
end