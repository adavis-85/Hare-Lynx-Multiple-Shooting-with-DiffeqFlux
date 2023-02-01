using Pkg

Pkg.add("DiffEqFlux")
Pkg.add("StochasticDiffEq")
Pkg.add("Plots")
Pkg.add("DiffEqBase")
Pkg.add("Statistics")
Pkg.add("DiffEqSensitivity")
Pkg.add("RecursiveArrayTools")
Pkg.add("Flux")
Pkg.add("Plots")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Optim")
Pkg.add("DifferentialEquations")
Pkg.add("Optimization")
Pkg.add("OptimizationFlux")
Pkg.add("Interpolations")
Pkg.add("Statistics")
Pkg.add("Distributions")
Pkg.add("OptimizationPolyalgorithms")
Pkg.add("Lux")
Pkg.add("Random")

using Flux, DiffEqFlux, StochasticDiffEq, Plots, DiffEqBase.EnsembleAnalysis, Optimization, OptimizationFlux,
      Statistics, DiffEqSensitivity, RecursiveArrayTools, CSV, DataFrames, Optim, DifferentialEquations,Interpolations,Distributions
using Base.Iterators: repeated
using OptimizationPolyalgorithms
using DiffEqFlux: group_ranges
using Lux
using Random

##uploading data
dataset=CSV.read("C:/Users/User 1/Downloads/hare_lynx.csv", DataFrame)

##permuting as there are only two variables needed Hare and Lynx yearly population
original_data = permutedims(Array{Float64}(dataset[:,2:3]))

############################################## DATA CONVERSION ############################################################
##Fully smoothed set to be trained on 66% of the data
original_data = log.(original_data)
original_data = original_data .- mean(original_data)
L=length(original_data[1,:])
##quarterly (three months) steps for smoothing
original_datasize = L
original_tspan = (0.0f0, Float32(L-1)) # Time range
original_tsteps = range(original_tspan[1], original_tspan[2], length = original_datasize)
hspline = CubicSplineInterpolation(original_tsteps,original_data[1,:])
lspline = CubicSplineInterpolation(original_tsteps,original_data[2,:])

##optional gaussian smoothing here
original_tsteps = range(original_tspan[1], original_tspan[2], length = 1+4*(original_datasize-1)) # Split to equal steps
original_data=vcat(hspline.(original_tsteps)',lspline.(original_tsteps)')

t_span_end=Int(floor(.66*L))
datasize=Int(floor(.66*length(original_data[1,:])))
ode_data=original_data[:,1:datasize]
tspan = (0.0f0, Float32(t_span_end-1)) # Time range
tsteps = range(tspan[1], tspan[2], length = datasize)

u0 = ode_data[:,1] 

################################################# NEURALNETWORK ###########################################################

rng = Random.default_rng()

nn = Lux.Chain(x -> x.^3,
                  Lux.Dense(2, 16, sigmoid),
                  Lux.Dense(16, 2))
p_init, st = Lux.setup(rng, nn)

neuralode = NeuralODE(nn, tspan, Tsit5(), saveat = tsteps)
prob_node = ODEProblem((u,p,t)->nn(u,p,st)[1], u0, tspan, Lux.ComponentArray(p_init))

############################################### LOSS AND CALLBACK FUNCTIONS ###############################################

function plot_multiple_shoot(plt, preds, group_size)
    step = group_size-1
    ranges = group_ranges(datasize, group_size)

    ##works
    for (i, rg) in enumerate(ranges)
        plot!(plt, tsteps[rg], preds[i][1,:], markershape=:circle, label="Group $(i)",legend=false)
    end
end

##needed to input all the losses from training for visualization
losses=[]
##it is possible that the optimal is met and crossed over and is accessible by pred_list
pred_list=[]
##option to save animation gif later on
anim = Plots.Animation()

iter = 0
callback = function (p, l, preds; doplot = true)
  display(l)
  push!(losses,l)
  push!(pred_list,preds)
  global iter
  iter += 1
  if doplot 
        # plot the original data
        plt = scatter(tsteps, ode_data[1,:], label = "Data",legend=false)

        # plot the different predictions for individual shoot
        plot_multiple_shoot(plt, preds, group_size)
        
        ##needed for gif of plots of networks and data
        frame(anim)
        
    if iter%100 == 0
        display(plot(plt))
    end
  end
  return false
end

##group size for each neural network and penalty for continuity.  the larger the group size the smaller the continuity_term
##the smaller the group_size, the larger the continuity_term.  
group_size = 15
continuity_term = 14


function loss_function(data, pred)
    return sum(abs2,(data .-pred))
end

function loss_multiple_shooting(p)
    return multiple_shoot(p, ode_data, tsteps, prob_node, loss_function, AutoTsit5(Rosenbrock23()),
                          group_size; continuity_term)
end

################################################### OPTIMIZATION FUNCTIONS ################################################
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, Lux.ComponentArray(p_init))
res_ms = Optimization.solve(optprob, PolyOpt(),
                                callback = callback)


################################################## OPTION FOR RETRAINING ##################################################

##if this is run repeatedly it will adjust the accuracy of the model. each res_ms is updated each time that the
##solver converges
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, res_ms.minimizer)
res_ms = Optimization.solve(optprob, PolyOpt(),
                                callback = callback)

##saves gif of plot and data on environment
##gif(anim, "multiple_shooting_total_training.gif", fps=15)

################################################## INDEXING OPTIMAL DATA ##################################################
best_spot=argmin(losses)

final=preds[best_spot][1][:,1:end-1]

##chooses the closest prediction made at each overlapping point
for i in 2:length(preds[best_spot])
    first_overlap=preds[best_spot][i-1][:,end]
    second_overlap=preds[best_spot][i][:,1]
    data_point=ode_data[:,(length(final[1,:]))+1]
    first_distance=sum(abs2,first_overlap-data_point)
    second_distance=sum(abs2,second_overlap-data_point)
    if first_distance<second_distance
        final=hcat(final,first_overlap,preds[best_spot][i][:,2:end-1])
    else
        final=hcat(final,second_overlap,preds[best_spot][i][:,2:end-1])
    end
end

##final prediction
final=hcat(final,preds[best_spot][end][:,end])

############################################### GRAPHS ####################################################################


plt2 = Plots.plot(tsteps, ode_data[1,:],
                     label = "data", title = "Neural ODE: Hare After Training",
                     xlabel = "Time")
plot!(plt2,tsteps, final[1,:], lw = 4, label = "prediction")

plt2 = Plots.plot(tsteps, ode_data[2,:],
                     label = "data", title = "Neural ODE: Lynx After Training",
                     xlabel = "Time")
plot!(plt2,tsteps, final[2,:], lw = 4, label = "prediction")

############################################# SOLVING PROBLEM OF ODE TO PREDICT TEST DATA #################################

set_datasize = length(original_data[1,:])-length(ode_data[1,:]) # Number of time points
set_tspan = (60.0f0, 90.f0) # Time range
set_tsteps = range(set_tspan[1], set_tspan[2], length = set_datasize)

prob_node = ODEProblem((u,p,t)->nn(u,p,st)[1], original_data[:,(datasize+1)], set_tspan, res_ms.minimizer)
sol=(solve(prob_node,Tsit5(), saveat = set_tsteps))
new_preds=hcat(sol.u...)

############################################# RESULTING PLOTS #############################################################
##plot of relation of predicted data Hare/Lynx
plt2 = Plots.plot(sol.t, new_preds[1,:],
                     label = "lynx prediction", title = "Model prediction",
                     xlabel = "Time")
plot!(plt2,sol.t, new_preds[2,:], lw = 2, label = "hare prediction")

##########################################################
##Lynx predicted data against the original data test set
start_spot=datasize+1
plt2 = Plots.plot(set_tsteps, original_data[2,start_spot:end],
                     label = "data", title = "Lynx Data",
                     xlabel = "Time")
plot!(plt2,sol.t, new_preds[2,:], lw = 4, label = "prediction")

##########################################################
##Hare predicted data against the original data test set
plt2 = Plots.plot(set_tsteps, original_data[1,start_spot:end],
                     label = "data", title = "Hare Data",
                     xlabel = "Time")
plot!(plt2,sol.t, new_preds[1,:], lw = 4, label = "prediction")

##########################################################
##trained Hare with hare predictions.  combining the two
plt2 = Plots.plot(tsteps, ode_data[1,:],
                     label = "data", 
                     xlabel = "Time")
plot!(plt2,tsteps, final[1,:], lw = 4, label = "Trained Prediction")
plot!(set_tsteps, original_data[1,start_spot:end],
                     label = "Test Data", title = "Hare Data Trained and Predicted",
                     xlabel = "Time")
plot!(sol.t, new_preds[1,:], lw = 4, label = "Forecast Prediction")
vline!([60],lw = 4,legend=false) 

##########################################################
##trained Lynx with hare predictions.  combine the two
plt2 = Plots.plot(tsteps, ode_data[2,:],
                     label = "data", 
                     xlabel = "Time")
plot!(plt2,tsteps, final[2,:], lw = 4, label = "Trained Prediction")
plot!(set_tsteps, original_data[2,start_spot:end],
                     label = "data", title = "Lynx Data Trained and Predicted",
                     xlabel = "Time")
plot!(sol.t, new_preds[2,:], lw = 4, label = "Forecast Prediction")
vline!([60],lw = 4,legend=false) 


