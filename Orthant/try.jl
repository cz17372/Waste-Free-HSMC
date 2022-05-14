using Distributions,Plots,LinearAlgebra,StatsPlots
x = rand(Normal(0,10),2,2)
M = transpose(x)*x
d = MultivariateNormal(zeros(2),M)
invd = MultivariateNormal(zeros(2),invM)
invM = inv(M)
true_sample = rand(d,10000)

HMC_sample = zeros(2,10000)
HMC_sample[:,1] = [0,0]
t = pi/2
for n = 2:10000
    v = rand(invd)
    HMC_sample[:,n] = HMC_sample[:,n-1]*cos(t) .+ M*v*sin(t)
end

density(HMC_sample[1,:])
density!(true_sample[1,:])


x = rand(Normal(0,10),4,4)
M = transpose(x)*x 
cholesky(M).L
cholesky(M[1:3,1:3]).L