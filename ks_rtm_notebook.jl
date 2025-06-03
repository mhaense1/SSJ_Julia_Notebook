### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 478e4840-348f-11f0-38c2-d76b6a6ff32f
begin
#loading Julia Packages that will be used below
using Parameters, LinearAlgebra, SparseArrays
using Accessors, BasicInterpolators, NonlinearSolve
using Statistics
using Plots
using PlutoUI: TableOfContents
end

# ╔═╡ 8936bc61-2880-4cb8-acc5-17a6526d82c5
md"""

# Global Solutions of Heterogeneous Agent Models in the Sequence Space - the Repeated Transition Method (Lee, 2025)

## by Matthias Hänsel

In two previous notebooks (available [here](https://mhaense1.github.io/SSJ_Julia_Notebook/SSJ_notebook.html) and [here](https://mhaense1.github.io/SSJ_Julia_Notebook/SSJ_notebook_2.html)), I provided Julia code and some explanations on how to obtain first-order (linearized) solution of Heterogeneous Agent (HA) DSGE models in the Sequence Space. 

While linearization is easily (and quickly) applicable to a wide range of models, it comes with two important downsides: 
Firstly, the quality of linearized solutions decreases once the model economy moves away from the point of approximation (how much can depend a lot on the model). Secondly, everything is certainly equivalent, i.e. the variance of aggregate shocks does not affect the model solution (except for the scale of the responses, of course).

For some applications that isn't much of a problem, while for others it is:
For example, if you are interested in how a higher or lower variance of aggregate shocks affects the macroeconomy, linearized model solutions won't be very useful.

In turn, you also would like to have methods that provide global solutions for HA models and thus don't suffer that much from the mentioned drawbacks. I recently taught myself about such a method, the **Repeated Transition Method** (RTM) proposed in the working paper of *[Lee (2025)](https://hanbaeklee.github.io/Webpage/Lee_AggRepTrans_2025.pdf)*. And in this notebook, I wish to make my own implementation of that method available in case people find it helpful.

Note that Lee himself provides a suite of **[Matlab Codes](https://sites.google.com/view/hanbaeklee/computation-lab)** that implements his method for different types of HA and representative agent models. For learning about the method, you should check them out as well. Nevertheless, the codes in this notebook do not only provide an implementation in a different language but also seem to improve computational speed by roughly an order of magnitude. This is despite the code being written to provide for a ''clear'' implementation instead of one being hyper-optimized for speed (for example, you could still eliminate allocations at many places).

Finally, note that there are also other methods that help with the problems of linearized HA model solutions mentioned above. If only non-linearities are a concern (i.e. for studying the effects of a big policy shock that is not expected to be repeated), you might want to check out the working paper by *[Boehl (2025)](https://gregorboehl.com/papers/hank_speed_boehl.pdf)* that proposes a method to reliably compue non-linear perfect foresight transition paths. If aggregate risk matters too, you could also explore higher-order perturbation methods (e.g. by *[Bhandari et al. (2023)](https://www.nber.org/system/files/working_papers/w31744/w31744.pdf)*) or use an approach a la *Krusell and Smith (1998)*.
"""

# ╔═╡ 3ccd44bb-e3e6-4318-ac7f-2047737c9842
TableOfContents()

# ╔═╡ c05980ac-7862-4dee-8830-bd07d2ed63d9
md""" 
## The model

As an example model, I am going to use the model described by *[Den Haan et al. (2010)](https://www.sciencedirect.com/science/article/pii/S0165188909001286)*, which was used an example environment for the 2010 special issue of the *Journal of Economic Dynamics and Control* on solving HA business cycle models. A brief outline is provided below for your convenience.

### Production 

* Production is done by a neo-classical firm sector employing a Cobb-Douglas technology of form $Y_t = Z_t K_t^\alpha L_t^{1-\alpha}$, where $Z_t$ is TFP.
Capital $K_t$ depreciates by $\delta$ every period. Under the standard assumptions, firms input choices will be characterized by 
```math
r_t = \alpha Z_t \left(\frac{K_t}{L_t}\right)^{\alpha-1} ~~\text{and}~~w_t = (1-\alpha) Z_t \left(\frac{K_t}{L_t}\right)^{\alpha} 
```
TFP $Z_t$ is exogenously time-varing. Specifically, it is assumed that the economy switches between a boom state in which $Z_t = 1+\Delta_Z$ and a recession state in which $Z_t = 1-\Delta_Z$. The aggregate state follows a two-state Markov process with transition matrix $\Pi_A$.

### Households

* There is a unit mass of ex-ante identical households who differ ex-post by their income state $z$ (''employment state'') and asset holdings $a$. 

* The employment state $z$ can take on one of two values $1$ and $0$, indicating being employed or unempoyed respectively. Employed workers provide $\mathcal{l}$ units of labor, each of which is compensated at wage rate $w_t$. Their income is furthermore taxed by the government at rate $\tau_t$. Unemployed workers receive Unemployment Insurance (UI) benefits according to a fixed replacement rate $\mu$.

* Savings choices $a$ are the solution to the household's consumption saving problem, which is characterized by the following Bellman equation
```math
V(z,a, \Gamma_t) = \max_{a'} \frac{c^{1-\sigma}-1}{1-\sigma} + \beta \mathbb{E}_t\left[\sum_{z'\in \lbrace 0,1 \rbrace}P(z'\vert z, Z')V(z',a',\Gamma_{t+1})\right]
```
```math
\text{subject to}~~c+a' = (1+r_t-δ)a + (1-\tau_t)z\mathcal{l}w_t + (1-z)\mu w_t ~~\text{and}~~a'\geq \bar{a}
```

* The employment transition probabilities are assumed to depend on the realized aggregate state next period $Z$ (e.g., unemployment risk is higher in recessions). $\Gamma$ collects all aggregate state variables relevant for the household, in particular $Z_t$ and the wealth distribution.

* We denote by $D$ the distribution of households, i.e. $D_t(s,a)$ is the mass of households who start the period with skill $s$ and assets $a$.

* For convenience, I will also define $\omega_t(z)$ to be the mass of households who have income state $z\in \lbrace 0,1 \rbrace$ in period $t$. It evolves according to
```math
\omega_t(z) = \sum_{z_{-1} \in\lbrace 0,1 \rbrace}P(z \vert z_{-1} ,Z_t)\omega_{t-1}(z_{-1}).
```

### Government

The government runs a balanced budget and always sets the tax rates accordingly, i.e.
```math
\tau_t w_t L_t = \mu w_t \omega_t(0) \implies \tau_t = \mu \frac{\omega_t(0)}{L_t}.
```

### Market clearing

In an equilibrium, the following market clearing conditions will have to hold:

* Asset market:
```math
K_t = \sum_{z\in \lbrace 0,1 \rbrace}\int a D(s,a)da
```

* Labor market:
```math
L_t = \sum_{z\in \lbrace 0,1 \rbrace}\omega_t(z)z\mathcal{l}
```

* Goods market:
```math
Y_t = K_{t+1} - (1-\delta)K_t + \underbrace{\sum_{z\in\lbrace 0,1 \rbrace}\int c(s,a) D(s,a)da}_{=\text{aggregate consumption}} 
```

In this model, labor market clearing is trivial as there is no labor supply choice on behalf of hosueholds. Also, if the asset market clears, the goods market will also due to Walras' law.


### Calibration

The calibration follows [Den Haan et al. (2010)](https://www.sciencedirect.com/science/article/pii/S0165188909001286) and is displayed in the code block below:

"""

# ╔═╡ 94a20e6c-d019-4e84-8f7c-832250cb503b
@with_kw struct ModelParameters{T}

 #household parameters
 β::T = 0.99
 σ::T = 1.0
 𝓁::T =  1/0.9 #normalization labor productivity

 #production parameters
 α::T = 0.36
 δ::T = 0.025

 #policy
 μ::T = 0.15 

 #business cycle
 ΔA::T = 0.01  #productivity in boom/recession state

end

# ╔═╡ af7d31c0-c829-478a-aff2-17c207c69a28
md"""
We also define the transition matrices in accordance with [Den Haan et al. (2010)](https://www.sciencedirect.com/science/article/pii/S0165188909001286):
"""

# ╔═╡ d9226fcc-f6ba-41a9-ac51-90bc5b217597
begin
#transition matrices from den Haan, Judd and Juillard (2010, JEDC)

#Transition matrix for TFP state (aggregate risk)
ΠA = [0.875 0.125; 0.125 0.875]

#This matrix provides for four states (income state times TFP state)
ΠEmp = [0.525 0.35 0.03125 0.09375;
		0.038889 0.836111 0.002083 0.122917;
		0.09375 0.03125 0.291667 0.583333;
		0.009115 0.115885 0.024306 0.850694]

#The matrix to use for computing an initial stationary equilibrium
ΠEmpSS = [0.6 0.4; 0.044445 0.955555]
end

# ╔═╡ a3c7ee9b-d604-43a4-958a-3dea02ff2e4e
md"""
Finally, we define a structure that holds the parameters, grids and transition matrices as well as some related numerical parameters. 

For the income grid, we follow [Maliar et al. (2010)](https://www.sciencedirect.com/science/article/pii/S0165188909001328) and use a grid that provides for the points being much closer spaced at lower wealth levels. In line with Lee (2025)'s Matlab codes, we'll use the same grid for the policy functions and the discretized wealth distribution. However, for accuracy reasons, it may be desirable to use a finer grid for the wealth distribution instead.
"""

# ╔═╡ cd5eeabc-d26d-4789-a69a-8bd5de473770
#generates grid as in Maliar et al. (2010)
function power_grid(amin,amax,na::Int;pwr = 7.0)
	x  = collect(range(0.0,0.5,na))
	y  = x.^pwr./maximum(x.^pwr);
	return amin.+(amax-amin).*y;
end

# ╔═╡ f220ee45-ff4b-485f-bc25-f24598b7f4cf
@with_kw struct NumericalParameters{T,I}

#instance of model parameters
mp::ModelParameters{T} = ModelParameters()

#income process
#use commented terms for case with time-varing labor supply
nz::I = 2
ΠEmp::Matrix{T} = [1.0 0.0 ; 0.0 1.0] #spaceholder
ΠEmpSS::Matrix{T} = [1.0 0.0; 0.0 1.0] #spaceholder

#business cycle 
A_vec::Vector{T} = [1-mp.ΔA, 1+mp.ΔA]
nA::I = length(A_vec)
ΠA::Matrix{T} =  [1.0 0.0; 0.0 1.0] #spaceholder
	
#wealth grid
na::I = 100
amin::T = 0.0
amax::T = 1000.0
a_grid::Vector{T} = power_grid(amin,amax,na) 

#tolerances and related
HHtol_SS::T = 1e-10
Ktol_path::T = 1e-6
T_path::I = 1100
T_burnin::I = 100
weight_a::T = 0.1 
weight_K::T = 0.1
	
end

# ╔═╡ 4e995b5b-427f-4c07-82a6-97ec356cbb10
md"""

## Solving the Steady State

The RTM requires an initial steady state (SS), which is computed in this section.
As in my previous notebooks, I use the Endogenous Grid Method (EGM) to solve for household prolicies given a set of aggregates ($w_t, r_t, \tau_t$).
The EGM was already used in my [previous notebook](https://mhaense1.github.io/SSJ_Julia_Notebook/SSJ_notebook.html) and thus I won't provide a lot of explanations here.

To implement it, the function below takes next period's expecteted marginal value functions and relevant aggregates and iterates backward on the Euler equation:
"""

# ╔═╡ 95be7081-5620-42e1-9f80-f3db9bed05b4
function EGM_update(EVa,r,𝓁_incs,np::NumericalParameters)

	@unpack mp, a_grid, na, nz, amin = np
	@unpack β, σ, δ = mp

	#invert Euler
	c_endog = (β.*EVa).^(-1/σ)

	#vector of labor/transfer incomes for different types
    a_endog = (c_endog .- 𝓁_incs' .+ a_grid)./(1.0 + r - δ)

	#below the points, borrowing constraints become binding
    constr = a_endog[1,:]

	#pre-allocate some arrays
    aPrime = zeros(eltype(a_endog),na,nz)

	@views for zi = 1:nz #loop over ind. state  

        #interpolate savings for unconstrained states
        itp_a = LinearInterpolator(a_endog[:,zi],a_grid,NoBoundaries()) 
			
        aPrime[:,zi] = itp_a.(a_grid)

		#apply borrowing constraint
		aPrime[a_grid .< constr[zi],zi] .= amin     
    end

	#back out consumption from budget constraint
	c = (1+r-δ).*a_grid .+ 𝓁_incs' .- aPrime

	return c, aPrime

end

# ╔═╡ a07a0555-c316-4f25-8e07-add4bdd859dd
md"""
To solve the SS HH problem, the function below keeps iterating on the above function for a given set of aggregates unil convergence:
"""

# ╔═╡ 6f9762fa-c346-4e39-8afe-8a78113f7cb1
function solve_EGM_SS(wSS,rSS,τSS,np::NumericalParameters;  
						max_iter::Int = 3000, print::Bool = false)

	@unpack mp = np 
	@unpack δ, σ = mp

	#steady state incomes
	incs = [mp.μ*wSS,(1-τSS)*wSS*mp.𝓁]

	#get initial guess
	c0 = 0.1 .+ 0.1*((1+rSS-mp.δ)*np.a_grid .+ incs')
	
	local a1

	#iterate until convergence
	dist = 1.0; iter = 0
	while (dist>np.HHtol_SS) & (iter < max_iter)

		#get marginal value function tomorrow
		EVa = (1.0+rSS-δ).*((c0.^(-σ)))*np.ΠEmpSS'

		#EGM step
		c1, a1 = EGM_update(EVa,rSS,incs,np)

		#compute distance
		dist = maximum(abs.(c1 .- c0))
		
		if print & (rem(iter,100) == 0.0)
			println("iteration: ",iter," current distance: ",dist)
		end
		
		#update
		c0 = copy(c1)
		
		iter += 1
	end
	
	#check convergence
	println("Done after ",iter," iterations, final distance ",dist)
	if iter == max_iter 
		@warn "Non-convergence of solve_EGM_SS"
	end
	
	return c0, a1
end

# ╔═╡ 833b5582-f1b1-44dc-beb9-60a21fbcc54c
md"""
Let's try whether everything works for the SS EGM:
"""

# ╔═╡ e377e101-692b-497e-995e-3baf1494f6a9
begin
#initialize and instance of NumericalParameters
np = NumericalParameters(ΠEmp = ΠEmp, ΠEmpSS = ΠEmpSS, ΠA = ΠA)

#check that SS function EGM function works for some arbitrary inputs
testEGM = solve_EGM_SS(1.0,0.005,0.0,np,print = true)
end

# ╔═╡ 2829f3ed-f4c8-4ce6-bb18-6cd6ee0ee43b
md"""
For getting the steady state distribution, I again use the Young (2010) histogram method. Since this was already discussed in a previous [notebook](https://mhaense1.github.io/SSJ_Julia_Notebook/SSJ_notebook.html), I don't go into details and the functions that build the necessary transition matrix and solve for the stationary distribution are relegated to the hidden code fields below (they are called `build_Λ` and `inv_dist`, respectively).
"""

# ╔═╡ f5711d64-d88d-46d2-9cac-5a74974d56a9
#function to build SS wealth & income histogram transition matrix
function build_Λ(a_choice::Matrix, ΠS::Matrix, np::NumericalParameters)

    @unpack  na, nz, a_grid = np

	#pre-allocate arrays
    weights_R = zeros(eltype(a_choice),na*nz,nz)
    weights_L = zeros(eltype(a_choice),na*nz,nz)
    IDX_col_R = zeros(Int64,na*nz,nz) #where agents "go"

    @views begin
        for ss = 1:nz
            for aa = 1:na

            #find closest smaller capital value on grid  
            al_idx = searchsortedlast(a_grid,a_choice[aa,ss])
    
            ss_shifter = (ss-1)*na #inner helper variable

                    #if clause checks for boundary cases
                    if al_idx == na   
					#if higher than highest grid point, I assign entire mass to maximum grid point	

                        for sss = 1:nz
                            weights_R[ss_shifter + aa,sss] += ΠS[ss,sss]
                            IDX_col_R[ss_shifter + aa,sss] = al_idx + (sss-1)*na
                        end

                    elseif al_idx == 0 
						#if lower than smallest grid point, I assign entire mass to lowest grid point	

                        for sss = 1:nz
                            weights_L[ss_shifter + aa,sss] += ΠS[ss,sss]
                            IDX_col_R[ss_shifter + aa,sss] = (sss-1)*na + 2
                        end

                    else    #regular case - using weights formula

                        wr = ((a_choice[aa,ss] - a_grid[al_idx]) / 
								(a_grid[al_idx+1] - a_grid[al_idx]))
                        lr = 1.0 - wr

                        for sss = 1:nz
                            weights_R[ss_shifter + aa,sss] += ΠS[ss,sss]*wr
                            weights_L[ss_shifter + aa,sss] += ΠS[ss,sss]*lr                         
                            IDX_col_R[ss_shifter + aa,sss] = (sss-1)*na + al_idx + 1
                        end

                    end

            end
        end

        IDX_from = repeat(1:(na*nz),outer=2*nz)
        weights = vcat(weights_R[:], weights_L[:])
        IDX_to = vcat(IDX_col_R[:],IDX_col_R[:] .- 1)
    end

    #return sparse transition matrix
    return sparse(IDX_from,IDX_to,weights,na*nz,na*nz)

end

# ╔═╡ d4f46f92-c225-46fd-aff9-c23424b86e02
#function to compute stationary distribution of a discrete markov chain
function inv_dist(Π::AbstractArray)
	#Π is a Stochastic Matrix
    x = [1; (I - Π'[2:end,2:end]) \ Vector(Π'[2:end,1])]
    return  x./sum(x) #normalize so that vector sums up to 1.
end

# ╔═╡ 5a7d8ca9-fa14-49f4-8e1f-ed16b15a2a13
md"""
Now, we are ready to solve for the economy's steady state equilibrium.
To do so, I use the two functions below which employ the Brent algorithm from the *[NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl)* package to find the interest rate $r_{ss}$ at which aggregate capital demand equals aggregate savings supply.
"""

# ╔═╡ eab4c524-5a81-4538-9070-fbb9822d1aea
begin
#functions to solve Steady State of Economy
function solve_SS(r_bracket, np::NumericalParameters; verbose::Bool = true)

	@unpack mp, ΠEmpSS, a_grid = np
	@unpack 𝓁, α, μ, β, δ = mp

	#get steady state income distribution
	incdist_SS = inv_dist(np.ΠEmpSS)

	#define objective for rootfinding
	#objective SS_objective is defined below
	obj_fun(rss,p) = SS_objective(rss,incdist_SS,np,verbose = verbose)
	prob_int = IntervalNonlinearProblem(obj_fun, r_bracket)

	#rootfinding using Brent's method
	sol = solve(prob_int,Brent())

	if sol.retcode != :Success
		@warn "Steady State didn't solve"
	end

	#back out SS objects
	r_ss = sol.u #solution of rootfinding process
	L_ss = incdist_SS[2]*𝓁

	KoverL = (r_ss/(α)).^(1/(α-1))

	w_ss = (1-α)*KoverL^α
	τ_ss = μ*incdist_SS[1]/L_ss

	c_ss, a_ss = solve_EGM_SS(w_ss,r_ss,τ_ss,np,print = false)
	Λ_ss = build_Λ(a_ss,ΠEmpSS,np)
	dist_ss = inv_dist(Λ_ss)

	K_ss = dist_ss'*repeat(a_grid,2)
	C_ss = dist_ss'*c_ss[:]
	Y_ss = K_ss^α * L_ss^(1-α) 

	clearing_check = Y_ss - δ*K_ss - C_ss

	println("market clearing residual: ",clearing_check)

	#return everything as named tuple
	return (; r_ss, w_ss,τ_ss, Y_ss, K_ss, L_ss, dist_ss, c_ss, a_ss)

end

function SS_objective(r_guess,incdist_SS,np::NumericalParameters; verbose::Bool = true)

	@unpack mp, ΠEmpSS, nz, a_grid = np
	@unpack 𝓁, α, μ = mp

	L = incdist_SS[2]*𝓁

	KoverL = (r_guess/(α)).^(1/(α-1))

	K_demand = KoverL*L
	w = (1-α)*KoverL^α
	τ = μ*incdist_SS[1]/L

	c_ss, a_ss = solve_EGM_SS(w,r_guess,τ,np,print = false)
	Λ_ss = build_Λ(a_ss,ΠEmpSS,np)
	dist = inv_dist(Λ_ss)

	K_supply = dist'*repeat(a_grid,2)

	dist = K_supply - K_demand

	if verbose
	println("Distance: ",dist," wage, tax, r: ",w," ",τ," ",r_guess)
	end

	return dist
	
end

end

# ╔═╡ 4aee11f0-6ab2-4088-8068-3e49158ad7c3
md"""
Let's get the steady state then. We can also time how long it takes.
"""

# ╔═╡ 230439df-c697-4b5e-acc3-b1ebc83be912
#get Steady State output - requires specifying intervall for SS interest rate
out = solve_SS((np.mp.δ,1/np.mp.β + np.mp.δ - 1.0),np)

# ╔═╡ c2b83e6c-a21a-466f-b587-bb9dc45c2909
#time it
@time solve_SS((np.mp.δ,1/np.mp.β + np.mp.δ - 1.0),np, verbose = false)

# ╔═╡ 854d53fe-0f01-4383-88e7-11ba2588603e
md"""
On my Dell XPS laptop with 11th Gen Intel(R) Core(TM) i7-11800H processor, getting the SS takes approximately 0.6 seconds once all the functions are already complied. 

Note that the computation time displayed in the HTML notebook is the one for the Github server running the uploaded notebook, which may not reflect well how long the code will take on a regular PC.
"""

# ╔═╡ cf446947-b34d-47cb-b466-3b9b4cc64ab3
md"""

## The Repeated Transition Method (RTM)

In this Section, I will implelemt the Repeated Transition Method (RTM). I was considering providing a detailed description of how it works here, but I was a bit too lazy and refer to Section 2 and 3 of *[Lee (2025)](https://hanbaeklee.github.io/Webpage/Lee_AggRepTrans_2025.pdf)* instead. Maybe I'll add the detailed explanation some day in the future. Until then, check the algorithm summary at the end of Section 2.3 in Lee (2025) to follow the code.

### Helper functions

We start with defining some functions that will be useful for simulating the model's stochastic path. First, a function that simulates a Markov Chain with transition matrix $Π$:
"""

# ╔═╡ fb75058c-c832-4e03-830a-5d058553bbd7
#function that simulates a markov chain for T periods
function simulate_markov(Π::Matrix,T::Int,init::Int)

	@assert init <= size(Π)[1]

	#get bin boundaries
	Πc = cumsum(Π,dims = 2)

	#uniform draw
	unif_draws = rand(T)

	#containers
	states = zeros(eltype(init),T+1)
	states[1] = init

	for tt = 1:T #transition according to which bin the uniform draws end up in
		states[tt+1] = searchsortedfirst(Πc[states[tt],:],unif_draws[tt])
	end

	return states

end

# ╔═╡ 57fce5e3-4bb6-4c45-bdeb-4c1ee7236d44
md"""
Next, we add a function that extracts household employment transition probabilities conditional on the aggregate states from `ΠEmp`.
"""

# ╔═╡ 782c9075-aa13-4ec8-b10c-4a9c9a3340e6
#The RTM as implemented by Lee (2025) exploits two features of common HA DSGE models:

#- If you simulate them long enough, you will eventually visit all relevant regions of the state space.

#- You can typically find a small vector of ''sufficient statistics'' $e_t$ that, together with the exogenous shock process, provide sufficient information to characterize the state of the economy. In particular, that means that for a given state of the exogneous process, just comparing the ''sufficient statistics'' tells you how similar the state of the economy is. See the paper for a more formal definition.

#Below, we shall assume that there is a unidimensional sufficient statistic $e_t$. This is the case for the models considered by Lee, but may not hold for any model, even though there is freedom on how to choose/construct $e_t$.

#To solve the model, the RTM following Lee (2025) then requires us to do the following: 

#1. We fix a long path of aggregate shocks $Z_t$ and a guess of the corresponding evolution of $e_t$ (e.g., its steady state value plus some noise).

#2. We start with some initial policy- and/or (marginal) value functions for each period along the simulation path. These could be the steady state policy functions.

#4. In any period along the path, we 

# ╔═╡ b6173493-ce2e-4653-a89d-c6d4eaf32fd2
#function that extract business cycle dependent transition matrices from ΠEmp
function get_ΠEmp_by_A(np::NumericalParameters)

	@unpack nz, nA, ΠEmp, ΠA = np
	@assert size(ΠEmp) == (nz*nA, nz*nA) "ΠEmp has inconsistent size"

	#container
	ΠEmp_by_A = Matrix{Matrix{Float64}}(undef,nA,nA)

	#extract and save in container
	for AA0 = 1:nA #current A
		for AA1 = 1:nA #next A
		ΠEmp_by_A[AA0,AA1] = ΠEmp[(AA0-1)*nA .+ (1:nz), (AA1-1)*nA .+ (1:nz)]./ΠA[AA0,AA1]
		@assert all(sum(ΠEmp_by_A[AA0,AA1],dims=2) .≈ 1.0)
		end
	end

	return ΠEmp_by_A

end

# ╔═╡ d48632dd-57d0-4eec-a0cb-c715beb90c3c
md"""
With these aggregate state-dependent transition matrices, it is more easy to simulate the evolution of the income distribution and in turn aggregate labor supply, which the function below does.
"""

# ╔═╡ 5a02d4e7-b86a-4531-a87e-f825199ab7da
#function that simulates the exogenously evolving income distribution for T periods
#(for given path of aggregate states)
function simulate_incs(ΠEmp_by_A,states::Vector{Int}, init_dist::Vector)

	 #Note: assumes that for each state there is a separate transition matrix saved in ΠEmp_by_A
	 T = length(states) #length of path

	#initialize conatiners
	 dist_path = zeros(length(init_dist),T)
	 dist_path[:,1] = init_dist

	#apply income transitions
	 for tt = 1:(T-1)
		@views dist_path[:,tt+1] = ΠEmp_by_A[states[tt],states[tt+1]]'*dist_path[:,tt]
	 end
	
	 return dist_path
end

# ╔═╡ dc8dc412-5f89-4389-ae0d-91bcb8204827
md"""
Finally, we add a function that, given a grid `grid` and value `val`, provides the weights to linearily interpolate a function known on the grid at the value.
"""

# ╔═╡ d25e8698-0625-435c-a78f-4b6b73d368c8
begin
#function that obtains linear interpolation weights for value val 
#assumes grid is sorted in increasing fashion
function grid_weights(grid::Vector,val,max_idx::Int)

	#find closest neighbour from below
	val_idx = searchsortedlast(grid,val)

	#if clause checks for boundary cases
	if (val_idx == max_idx)   
		#if higher than highest grid point, I assign entire mass to maximum grid point, i.e, no extrapolation
		val_idx = max_idx-1
		weight_l = 0.0 ; weight_r = 1.0

	elseif val_idx == 0 
		#if lower than smallest grid point, I assign entire mass to lowest grid point, i.e, no extrapolation
		val_idx = 1
		weight_l = 1.0 ; weight_r = 0.0
	else
		#return standard interpolation weights
		weight_r = ((val - grid[val_idx]) / (grid[val_idx+1] - grid[val_idx]))
		weight_l = 1.0 - weight_r
	end

	return weight_r, weight_l, val_idx

end

#convenience method that takes the max index as the upper end of the grid
function grid_weights(grid::Vector,val)
	return  grid_weights(grid,val,length(grid))
end

end

# ╔═╡ 214baa0c-c053-47ca-80d1-0139f3c8bd83
md"""

### The Backward Solution

The function below conducts the Backward Solution Step of the RTM, using the aggregate capital stock as sufficient statistic to interpolate household expected marginal value functions in every period.

To update households' value and policy functions, I again use the EGM, recycling the function `EGM_update` I defined above. 

"""

# ╔═╡ c2c88bf6-891a-4025-91f4-915ab4a1af96
function backward_solution!(a_pols_new, c_pols_new,a_pols,
							 aggr_states::Vector{Int},K_aggr,inc_dists,
								ΠEmp_by_A,np::NumericalParameters)

	@unpack mp, T_path, T_burnin, A_vec, nz, na, nA, a_grid,  ΠA, amin = np
	@unpack 𝓁, α, δ, σ, μ = mp

	T_tot = T_path + T_burnin

    #to easily interpolate relevant policy functions in terms of 
	#sufficient statistic below
	SStat_sortedby_A = Vector{Vector{Float64}}(undef,nA)
	SStat_locs_byA = Vector{Vector{Int64}}(undef,nA)
	for AA in 1:nA
		SStats = K_aggr[aggr_states .== AA]
		SStat_loc = findall(x -> x == AA,aggr_states)

		SStats = SStats[(SStat_loc .< T_path) .& (SStat_loc .> T_burnin)]
		SStat_loc = SStat_loc[(SStat_loc .< T_path) .& (SStat_loc .> T_burnin)]

		idx = sortperm(SStats)

		SStat_sortedby_A[AA] = SStats[idx] 
		SStat_locs_byA[AA] = SStat_loc[idx]
	end

	#go backward on stochastic path
	for tt in Iterators.reverse(1:T_tot)

		#aggregate economy in t
		A_idx = aggr_states[tt]
		L_t = inc_dists[2,tt]*𝓁
		A = A_vec[A_idx]
		KoverL_t= K_aggr[tt]/(L_t)
		r = α*A*KoverL_t^(α-1)
		w = (1-α)*A*KoverL_t^(α)
		τ = μ*(1-inc_dists[2,tt])/L_t ;

		EVa = zeros(na,nz)

		#get expectation for next period 
		for aa = 1:nA

			#(predicted) aggregates conditional on shock aa
			inc_distPrime = ΠEmp_by_A[A_idx,aa]'*inc_dists[:,tt]
			LPrime = inc_distPrime[2]*𝓁
			APrime = A_vec[aa]
			KoverLPrime = K_aggr[tt+1]/(LPrime)
			rPrime = α*APrime*KoverLPrime^(α-1)
			wPrime = (1-α)*APrime*KoverLPrime^(α)
			τPrime = μ*(1-inc_distPrime[2])/LPrime


			#get savings policy conditional on shock aa
			if (aa == aggr_states[tt+1]) && (tt<T_tot)

				#if ZPrime is on path, no need for interpolation
				aPrime = a_pols[:,tt+1]		

			else

				#get interpolation weights
				Kr_weight, Kl_weight, gr_idx = grid_weights(SStat_sortedby_A[aa],K_aggr[tt+1])

				#interpolate savings policies
				aPrime = Kl_weight*a_pols[:,SStat_locs_byA[aa][gr_idx]] .+ 
						Kr_weight*a_pols[:,SStat_locs_byA[aa][gr_idx+1]]
			
			end

			#enforce borrowinng constraint
			aPrime .= max.(aPrime, amin)

			#back out consumption
			cPrime = (1+rPrime-δ)*a_grid .+ [μ*wPrime,(1-τPrime)*wPrime*𝓁]' .-
						reshape(aPrime,(na,nz))

			#compute expected marginal value function
			EVa .+= ΠA[A_idx,aa]*(((1+rPrime-δ).*
						(cPrime.^(-σ)))*ΠEmp_by_A[A_idx,aa]')
			
		end

		#apply EGM step
		c1, a1 = EGM_update(EVa,r,[μ*w,(1-τ)*w*𝓁],np)

		#save results in supplied matrix
		a_pols_new[:,tt] .= a1[:]
		c_pols_new[:,tt] .= c1[:]
	end
end

# ╔═╡ efacf511-78cb-433b-ad3d-a8b08f483ec7
md"""

### The Forward Simulation

With the updated policy functions, we have to simulate the wealth distribution along the stochastic path. 

To that end, I first define the function `DirectTransition!` which takes in the savings policy function `a_choice` (on the grid) as well as the current distribution `distr_old` (represented as a histogram) and propagates it according to the Young (2010) method. Essentially, it applies the histogram transition matrix without actually constructing it.
"""

# ╔═╡ f9f75345-7fe2-429a-b5fa-8d8f5876d379
#function that updates the income & wealth histogram given policy functions and income transition matrix
function DirectTransition!(distr_new::AbstractVector,distr_old::AbstractVector,
							a_choice::Matrix,ΠS::Matrix, np::NumericalParameters)

	@unpack nz, na, a_grid = np

	distr_new .= zero(eltype(distr_new))

	@views begin
	for zz = 1:nz
		idx_old = (zz-1)*na
		tr_probs =  ΠS[zz,:]
		
		for aa = 1:na

			#transition weights
			weight_r, weight_l, al_idx = grid_weights(a_grid,a_choice[aa,zz],na)

			#propagate ditribution forwards
			for zzz = 1:nz
				idx_l = (zzz-1)*na + al_idx
				distr_new[idx_l] += tr_probs[zzz]*weight_l*distr_old[idx_old+aa]
				distr_new[idx_l+1] += tr_probs[zzz]*weight_r*distr_old[idx_old+aa]
			end

		end
  	end
	end

end

# ╔═╡ b2a6a10d-0397-4a2c-b41f-4b8e558ea720
md""" 

Let's briefly check that that works as it is supposed to: If yes, applying DirectTransition to the steady state histogram with the steady state policies should again return the steady state histogram.

"""

# ╔═╡ 63b8f98c-2af7-4395-b2f6-22cff535e5c5
begin
#check that DirectTransition! works (using SS policy functions)
tester = zeros(length(out.dist_ss))
DirectTransition!(tester,out.dist_ss,out.a_ss,np.ΠEmpSS,np)
maximum(abs.(tester.-out.dist_ss))
end

# ╔═╡ a02be7d1-179f-43e4-9166-2580f6168cd5
md"""
Reassured that this is the case, it is now easy to program up a function that simulates the wealth distribution along the simulation path and computes the various outcomes we are interested in, such as aggregate savings and consumption.
"""

# ╔═╡ 5cdf998e-ae4c-481b-b1a4-f74f6f3de85d
function forward_distribution(apols,c_pols,ΠEmp_by_A::Array,states::Vector{Int},
									distr_init::Vector,np::NumericalParameters)
	
	@unpack na, nz, T_path, T_burnin, a_grid = np

	#total number of periods to simulate
	T_tot = T_path + T_burnin

	#helper to easily compute aggregate capital
	agrid_stacked = repeat(a_grid,nz)
	
	#initialize arrays
	distr0 = deepcopy(distr_init)
	distr1 = similar(distr0)
	K_aggr = zeros(T_tot+1) ; C_aggr=zeros(T_tot)
	K_aggr[1] = distr0'*agrid_stacked
	L_aggr = similar(C_aggr)

	#iterate distribution forward
	for tt = 1:T_tot

		#compute aggregates
		C_aggr[tt] = distr0'*c_pols[:,tt]
		L_aggr[tt] = sum(distr0[(na+1):end])

		#apply transition operator
		DirectTransition!(distr1,distr0,
				reshape(apols[:,tt],(na,nz)),ΠEmp_by_A[states[tt],states[tt+1]],np)

		#next period aggregate capital stock
		K_aggr[tt+1] = distr1'*agrid_stacked

		distr0 .= distr1 #update
	end

	return K_aggr, C_aggr, L_aggr
end

# ╔═╡ fee67b8a-31f5-4fa8-8a5f-36ec7ea58e43
md"""

### Putting things together

With all these components, it is now easy to put the entire RTM algorithm together.
This is done with the below function `simulation_path`: For the current path of the sufficient statistics (the aggregate capital stock) and policy functions, it conducts the backward solution and forward simulation and updates everything, again and again until convergence.

"""

# ╔═╡ 7fdd9fd6-1f59-440a-bf44-46259ed78838
#this function implements the RTM
function simulation_path(K_init::AbstractVector,aggr_states::Vector{Int},
				SS_obj::NamedTuple,np::NumericalParameters; 
				max_iter::Int = 500, verbose::Bool = false)

	@unpack mp, T_path, T_burnin, Ktol_path = np
	@unpack K_ss = SS_obj

	T_tot = T_path + T_burnin

	#pre-allocate arrays for micro_level policies
	a_pols = repeat(SS_obj.a_ss[:],1,T_tot)
	c_pols = repeat(SS_obj.c_ss[:],1,T_tot)
	a_pols_new = zeros(size(a_pols))
	K_aggr = zeros(T_tot+1,max_iter)
	K_pred = zeros(length(K_aggr),max_iter) #keeps track of predicted K paths
	C_aggr = zeros(T_tot)
	
	#start with provided capital path K_init
	K_aggr = deepcopy(K_init)
	K_pred = zeros(length(K_aggr),max_iter)
	K_pred[:,1]	= K_init
	#helper variable for handling income dists
	ΠEmp_by_A = get_ΠEmp_by_A(np)

	#path of aggregate labor supply
	inc_dists = simulate_incs(ΠEmp_by_A,aggr_states,inv_dist(np.ΠEmpSS))

	iter = 1 ; dist = 100.0
	while (dist > Ktol_path) && (iter < max_iter)

		#HH backward iteration on stochastic path
		#(function defined above)
		backward_solution!(a_pols_new,c_pols,a_pols,aggr_states,
							K_aggr,inc_dists,ΠEmp_by_A,np)

		#propagates distribution forwards and computes means
		#(function defined above)
		K_aggr1, C_aggr1, L_aggr1 = forward_distribution(a_pols_new,c_pols,ΠEmp_by_A,
											aggr_states,SS_obj.dist_ss,np)
		#check that labor supply paths don't change
		@assert all(L_aggr1 .≈ inc_dists[2,1:T_tot])
		
		#compute distcances
		dist_K = maximum(((K_aggr.-K_aggr1)[(T_burnin+1):T_tot]).^2)
		dist_C = maximum(((C_aggr.-C_aggr1)[(T_burnin+1):T_tot]).^2)
		dist = dist_K + dist_C

		if (rem(iter,20) == 0.0) && verbose #print info
		println("Iteration ",iter," Current dists: ",dist_K," ",dist_C)
		end

		if (iter<=(max_iter-1)) & (dist > Ktol_path) #don't update at last iteration
		K_aggr .= (1-np.weight_K).*K_aggr .+ np.weight_K.*K_aggr1
		a_pols .= (1-np.weight_a).*a_pols .+ np.weight_a.*a_pols_new
		end

		C_aggr .= C_aggr1
		K_pred[:,iter+1] .= K_aggr1

		iter+=1
	end

	if iter == max_iter
		@warn "Simulation path didn't converge"
	else
		println("Finished RTM after ",iter," iterations. Final dist: ",dist)
	end

	return C_aggr, K_aggr, K_pred[:,1:iter], a_pols, c_pols, inc_dists

end

# ╔═╡ de8d14bd-9ab9-4a7b-9071-eea87cf430e1
md"""
Using `simulation_path` requires supplying the exogenous path of aggregate productivity and an initial guess for the path of the capital stock, which we get below:
"""

# ╔═╡ 3a0ec6de-035b-4b45-891c-97a3486217b7
begin 
#get path of business cycle states
aggr_states = simulate_markov(np.ΠA,np.T_burnin+np.T_path,1)

#get guess for capital path (slight perturbation of SS value)
K_init = vcat(out.K_ss,ones(np.T_burnin+np.T_path)*out.K_ss .+ 0.001*(rand(np.T_burnin+np.T_path) .- 0.5))
end

# ╔═╡ be9dfb11-5cc7-4e92-a47f-95d65313efbf
md"""
Now, let's see whether everything converges..
"""

# ╔═╡ 7680f004-d427-42b3-9977-a7104b40adb8
#apply RTM (displaying iteration information)
out_sim = simulation_path(K_init,aggr_states,out,np,max_iter = 1000,verbose = true)

# ╔═╡ 3e544dca-976a-459f-ba76-0d57ba11fedf
md"""
It does! We can also check how long the alogrithm takes to solve the model:
"""

# ╔═╡ ab737a60-f13d-4c61-9d78-742b1baef1af
begin
#new shocks and initial capital series
aggr_states2 = simulate_markov(np.ΠA,np.T_burnin+np.T_path,1)
K_init2 = vcat(out.K_ss,ones(np.T_burnin+np.T_path)*out.K_ss .+ 0.01*(rand(np.T_burnin+np.T_path) .- 0.5))

#time it
@time out_sim2 = simulation_path(K_init2,aggr_states2,out,np,max_iter = 1000,verbose = false)
end

# ╔═╡ 9e1133a3-90d3-4868-8313-7e66a1b489a3
md"""

On my laptop, this takes roughly 15-17 seconds. Again, the time displayed in the uploaded version will not match that as the code is run on the Github server.

Actually, we can get it to be even faster if we use larger updating weights:

"""

# ╔═╡ f10678f6-5e94-4142-b6a8-f96c2cfefdb8
begin
np2 = @set np.weight_K = 0.3
np2 = @set np2.weight_a = 0.3

#time it
@time simulation_path(K_init2,aggr_states2,out,np2,max_iter = 1000,verbose = false)
end

# ╔═╡ b0467ac4-1f11-421d-ac22-3022b7b5f8c8
md"""

So, we can globally solve this example economy in less than 10 seconds (on my laptop)! However, a downside of higher weights is that they may eventually impede instead of speed up convergence.

### Accuracy checks

Finally, let's do some simple accuracy checks.

Given that consistently matching the dynamics of the aggregate capital stock with household savings was one of our convergence criteria, we naturally achieve a high degree of accuracy in this regard:

"""

# ╔═╡ d9c607a1-dc4c-4a65-94bc-3255e965d970
begin
println("Dynamic consistency report:")
println("max. abs. error (% of K_ss): ",maximum(100*abs.(out_sim2[2].-out_sim2[3][:,end])./out.K_ss)," %")
println("mean. squared error (% of K_ss): ",(100*sqrt(sum((out_sim2[2].-out_sim2[3][:,end]).^2)./length(out_sim2[2]))./(out.K_ss))," %")
end

# ╔═╡ 6780ee26-73f8-4aaa-8c7a-c8b2363c3dc6
md"""
Additionally, we should check whether micro-level policies converged accurately as well. 

For this purpose, the below function computes on-grid savings policy errors along the stochastic path, which mirrors the respective accuracy check for consumption policies performed in Lee (2025)'s Matlab codes.
"""

# ╔═╡ 250b1666-173b-4308-873d-0ffc725f7f46
function savings_errors(c_pols,a_pols, K_aggr, states,
								SS_obj::NamedTuple,np::NumericalParameters)

	@unpack mp, T_path, T_burnin, Ktol_path, na, nz = np

	T_tot = T_path + T_burnin

	#pre-allocate arrays for micro-level policies
	a_pols_check = zeros(size(a_pols))
	c_pols_check = zeros(size(a_pols))

	#helper variable for handling income dists
	ΠEmp_by_A = get_ΠEmp_by_A(np)

	#path of aggregate labor supply
	inc_dists = simulate_incs(ΠEmp_by_A,states,inv_dist(np.ΠEmpSS))

	#HH backward iteration on stochastic path
	backward_solution!(a_pols_check,c_pols_check,a_pols,states,
							K_aggr,inc_dists,ΠEmp_by_A,np)

	#absolute savings error checks whether a_pols and the sequence
	#of policy functions implied by it, a_pols_check, are close.
	a_errors = 100.0*abs.((a_pols .- a_pols_check)./c_pols)
	#I compute it as percent of individual's consumption
	#this avoids dividing by 0 as a_pols = 0 is possible
	
	#pre-allocate some objects
	distr0 = deepcopy(SS_obj.dist_ss)
	distr1 = similar(distr0)
	avg_a_error = zeros(T_tot); max_a_error = zeros(T_tot)

	#compute average and maximum consumption error in every period
	for tt = 1:T_tot

		avg_a_error[tt] = distr0'*a_errors[:,tt]
		max_a_error[tt] = maximum(a_errors[:,tt])

		#update distribution
		DirectTransition!(distr1,distr0,
				reshape(a_pols[:,tt],(na,nz)),ΠEmp_by_A[states[tt],states[tt+1]],np)

		distr0 .= distr1 #update distribution
	end
	
	#return errors for non-burnin periods
	return avg_a_error[(T_burnin+1):T_path], max_a_error[(T_burnin+1):T_path], 
				a_errors[:,(T_burnin+1):T_path]
	
end

# ╔═╡ 66bcf94e-c3fd-4a1c-a5fa-d1e8f4cd261e
begin
#conduct savings policy check
check_results = savings_errors(out_sim2[5],out_sim2[4],out_sim2[2],aggr_states2,
										out, np)

println("Error summary:")
println("Mean avg. savings error (% of cons.): ", mean(check_results[1]))
println("Maximum savings error  (% of cons.): ", maximum(check_results[3]))
end

# ╔═╡ 08e76add-9ff1-4b33-8303-bbfacccf33de
p1 = plot(1:(np.T_path-np.T_burnin),check_results[1],linewidth = 3.0,legend = :none,title = "Avg. saving error",xlabel = "Time", ylabel = "Avg. Error in % of cons.")

# ╔═╡ f2044bed-6181-4693-bf07-bd5270ee06e6
p2 = plot(1:(np.T_path-np.T_burnin),check_results[2],linewidth = 3.0, legend = :none,title = "Max. saving error",xlabel = "Time",ylabel = "Max. Error in % of cons.")

# ╔═╡ 0bf6cb63-6aab-454f-ba5c-2d7fd6e9f289
md"""

Both average and maximum savings policy errors are very small, indicating that individual policies converged well along the stochastic path. 

I leave more sophisticated off-grid accuracy checks as an exercise for the reader and stop here.

## Final Words

Thanks for reading! I hope you found this notebook useful. If yes, please consider starring its [Github repo](https://github.com/mhaense1/SSJ_Julia_Notebook) and share it with colleagues who might also find it useful. Also, feedback and suggestions are always welcome.

Finally, since you seem interested in Heterogeneous Agent Macro, have a look at my [Personal Website](https://mhaense1.github.io) to learn about my research in this area and/or to get in contact with me.

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Accessors = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
BasicInterpolators = "26cce99e-4866-4b6d-ab74-862489e035e0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NonlinearSolve = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Accessors = "~0.1.42"
BasicInterpolators = "~0.7.1"
NonlinearSolve = "~4.8.0"
Parameters = "~0.12.3"
Plots = "~1.40.11"
PlutoUI = "~0.7.23"
Statistics = "~1.11.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "07a2ff08093a5ee6e8d0c8de782155f7d30cd189"

[[deps.ADTypes]]
git-tree-sha1 = "e2478490447631aedba0823d4d7a80b2cc8cdb32"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.14.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "4e25216b8fea1908a0ce0f5d87368587899f75be"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BasicInterpolators]]
deps = ["LinearAlgebra", "Memoize", "Random"]
git-tree-sha1 = "3f7be532673fc4a22825e7884e9e0e876236b12a"
uuid = "26cce99e-4866-4b6d-ab74-862489e035e0"
version = "0.7.1"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.BracketingNonlinearSolve]]
deps = ["CommonSolve", "ConcreteStructs", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase"]
git-tree-sha1 = "637ebe439ba587828fd997b7810d8171eed2ea1b"
uuid = "70df07ce-3d50-431d-a3e7-ca6ddb60ac1e"
version = "1.2.0"
weakdeps = ["ForwardDiff"]

    [deps.BracketingNonlinearSolve.extensions]
    BracketingNonlinearSolveForwardDiffExt = "ForwardDiff"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2ac646d71d0d24b44f3f8c84da8c9f4d70fb67df"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.4+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "ConcreteStructs", "DataStructures", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "FastPower", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "Parameters", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "SciMLStructures", "Setfield", "Static", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "ae6f0576b4a99e1aab7fde7532efe7e47539b588"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.170.1"

    [deps.DiffEqBase.extensions]
    DiffEqBaseCUDAExt = "CUDA"
    DiffEqBaseChainRulesCoreExt = "ChainRulesCore"
    DiffEqBaseDistributionsExt = "Distributions"
    DiffEqBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    DiffEqBaseForwardDiffExt = ["ForwardDiff"]
    DiffEqBaseGTPSAExt = "GTPSA"
    DiffEqBaseGeneralizedGeneratedExt = "GeneralizedGenerated"
    DiffEqBaseMPIExt = "MPI"
    DiffEqBaseMeasurementsExt = "Measurements"
    DiffEqBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    DiffEqBaseReverseDiffExt = "ReverseDiff"
    DiffEqBaseSparseArraysExt = "SparseArrays"
    DiffEqBaseTrackerExt = "Tracker"
    DiffEqBaseUnitfulExt = "Unitful"

    [deps.DiffEqBase.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    GeneralizedGenerated = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "aa87a743e3778d35a950b76fbd2ae64f810a2bb3"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.6.52"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "e7b7e6f178525d17c720ab9c081e4ef04429f860"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.4"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.EnzymeCore]]
git-tree-sha1 = "0cdb7af5c39e92d78a0ee8d0a447d32f7593137e"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.8"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "Polyester", "Static", "StaticArrayInterface", "StrideArraysCore"]
git-tree-sha1 = "ab1b34570bcdf272899062e1a56285a53ecaae08"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.3.5"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastPower]]
git-tree-sha1 = "df32f07f373f06260cd6af5371385b5ef85dd762"
uuid = "a4df4552-cc26-4903-aec0-212e50a0e84b"
version = "1.1.2"

    [deps.FastPower.extensions]
    FastPowerEnzymeExt = "Enzyme"
    FastPowerForwardDiffExt = "ForwardDiff"
    FastPowerMeasurementsExt = "Measurements"
    FastPowerMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    FastPowerReverseDiffExt = "ReverseDiff"
    FastPowerTrackerExt = "Tracker"

    [deps.FastPower.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "f089ab1f834470c525562030c8cfde4025d5e915"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.27.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "301b5d5d731a0654825f1f2e906990f7141a106b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.16.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "910febccb28d493032495b7009dce7d7f7aee554"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.0.1"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "0ff136326605f8e06e9bcf085a356ab312eef18a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.13"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "9cb62849057df859575fc1dda1e91b82f8609709"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.13+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "1d4015b1eb6dc3be7e6c400fbd8042fe825a6bac"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.10"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "efadd12a94e5e73b7652479c2693cd394d684f95"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.10.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd714447457c660382fe634710fb56eb255ee42e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.6"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "866ce84b15e54d758c11946aacd4e5df0e60b7a3"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.6.1"

    [deps.LazyArrays.extensions]
    LazyArraysBandedMatricesExt = "BandedMatrices"
    LazyArraysBlockArraysExt = "BlockArrays"
    LazyArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    LazyArraysStaticArraysExt = "StaticArrays"

    [deps.LazyArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a31572773ac1b745e0343fe5e2c8ddda7a37e997"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "321ccef73a96ba828cd51f2ab5b9f917fa73945a"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.0+0"

[[deps.LineSearch]]
deps = ["ADTypes", "CommonSolve", "ConcreteStructs", "FastClosures", "LinearAlgebra", "MaybeInplace", "SciMLBase", "SciMLJacobianOperators", "StaticArraysCore"]
git-tree-sha1 = "97d502765cc5cf3a722120f50da03c2474efce04"
uuid = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
version = "0.1.4"

    [deps.LineSearch.extensions]
    LineSearchLineSearchesExt = "LineSearches"

    [deps.LineSearch.weakdeps]
    LineSearches = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LinearSolve]]
deps = ["ArrayInterface", "ChainRulesCore", "ConcreteStructs", "DocStringExtensions", "EnumX", "GPUArraysCore", "InteractiveUtils", "Krylov", "LazyArrays", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "Setfield", "StaticArraysCore", "UnPack"]
git-tree-sha1 = "1e1f3ba20d745a9ea57831b7f30e7b275731486e"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "3.9.0"

    [deps.LinearSolve.extensions]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCUDAExt = "CUDA"
    LinearSolveCUDSSExt = "CUDSS"
    LinearSolveEnzymeExt = "EnzymeCore"
    LinearSolveFastAlmostBandedMatricesExt = "FastAlmostBandedMatrices"
    LinearSolveFastLapackInterfaceExt = "FastLapackInterface"
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMetalExt = "Metal"
    LinearSolvePardisoExt = ["Pardiso", "SparseArrays"]
    LinearSolveRecursiveFactorizationExt = "RecursiveFactorization"
    LinearSolveSparseArraysExt = "SparseArrays"
    LinearSolveSparspakExt = ["SparseArrays", "Sparspak"]

    [deps.LinearSolve.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    FastLapackInterface = "29a986be-02c6-4525-aec4-84b980013641"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    RecursiveFactorization = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Sparspak = "e56a9233-b9d6-4f03-8d0f-1825330902ac"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "54e2fdc38130c05b42be423e90da3bade29b74bd"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.4"
weakdeps = ["SparseArrays"]

    [deps.MaybeInplace.extensions]
    MaybeInplaceSparseArraysExt = "SparseArrays"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "453de0fc2be3d11b9b93ca4d0fddd91196dcf1ed"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.5"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "BracketingNonlinearSolve", "CommonSolve", "ConcreteStructs", "DiffEqBase", "DifferentiationInterface", "FastClosures", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "LinearSolve", "NonlinearSolveBase", "NonlinearSolveFirstOrder", "NonlinearSolveQuasiNewton", "NonlinearSolveSpectralMethods", "PrecompileTools", "Preferences", "Reexport", "SciMLBase", "SimpleNonlinearSolve", "SparseArrays", "SparseMatrixColorings", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "7fd96e0e6585063a7193007349799155ba5a069f"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "4.8.0"

    [deps.NonlinearSolve.extensions]
    NonlinearSolveFastLevenbergMarquardtExt = "FastLevenbergMarquardt"
    NonlinearSolveFixedPointAccelerationExt = "FixedPointAcceleration"
    NonlinearSolveLeastSquaresOptimExt = "LeastSquaresOptim"
    NonlinearSolveMINPACKExt = "MINPACK"
    NonlinearSolveNLSolversExt = "NLSolvers"
    NonlinearSolveNLsolveExt = ["NLsolve", "LineSearches"]
    NonlinearSolvePETScExt = ["PETSc", "MPI"]
    NonlinearSolveSIAMFANLEquationsExt = "SIAMFANLEquations"
    NonlinearSolveSpeedMappingExt = "SpeedMapping"
    NonlinearSolveSundialsExt = "Sundials"

    [deps.NonlinearSolve.weakdeps]
    FastLevenbergMarquardt = "7a0df574-e128-4d35-8cbd-3d84502bf7ce"
    FixedPointAcceleration = "817d07cb-a79a-5c30-9a31-890123675176"
    LeastSquaresOptim = "0fc2ff8b-aaa3-5acd-a817-1944a5e08891"
    LineSearches = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
    MINPACK = "4854310b-de5a-5eb6-a2a5-c1dee2bd17f9"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NLSolvers = "337daf1e-9722-11e9-073e-8b9effe078ba"
    NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
    PETSc = "ace2c81b-2b5f-4b1e-a30d-d662738edfe0"
    SIAMFANLEquations = "084e46ad-d928-497d-ad5e-07fa361a48c4"
    SpeedMapping = "f1835b91-879b-4a3f-a438-e4baacf14412"
    Sundials = "c3572dad-4567-51f8-b174-8c6c989267f4"

[[deps.NonlinearSolveBase]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "CommonSolve", "Compat", "ConcreteStructs", "DifferentiationInterface", "EnzymeCore", "FastClosures", "LinearAlgebra", "Markdown", "MaybeInplace", "Preferences", "Printf", "RecursiveArrayTools", "SciMLBase", "SciMLJacobianOperators", "SciMLOperators", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "edfa90b9b46fc841b6f03106d9e1a054816f4f1d"
uuid = "be0214bd-f91f-a760-ac4e-3421ce2b2da0"
version = "1.6.0"

    [deps.NonlinearSolveBase.extensions]
    NonlinearSolveBaseBandedMatricesExt = "BandedMatrices"
    NonlinearSolveBaseDiffEqBaseExt = "DiffEqBase"
    NonlinearSolveBaseForwardDiffExt = "ForwardDiff"
    NonlinearSolveBaseLineSearchExt = "LineSearch"
    NonlinearSolveBaseLinearSolveExt = "LinearSolve"
    NonlinearSolveBaseSparseArraysExt = "SparseArrays"
    NonlinearSolveBaseSparseMatrixColoringsExt = "SparseMatrixColorings"

    [deps.NonlinearSolveBase.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LineSearch = "87fe0de2-c867-4266-b59a-2f0a94fc965b"
    LinearSolve = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"

[[deps.NonlinearSolveFirstOrder]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConcreteStructs", "DiffEqBase", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "LinearSolve", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLJacobianOperators", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "3a559775faab057f7824036c0bc5f30c74b00d1b"
uuid = "5959db7a-ea39-4486-b5fe-2dd0bf03d60d"
version = "1.4.0"

[[deps.NonlinearSolveQuasiNewton]]
deps = ["ArrayInterface", "CommonSolve", "ConcreteStructs", "DiffEqBase", "LinearAlgebra", "LinearSolve", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "SciMLOperators", "StaticArraysCore"]
git-tree-sha1 = "290d60e3e097eed44e0aba00643995a47284746b"
uuid = "9a2c21bd-3a47-402d-9113-8faf9a0ee114"
version = "1.3.0"
weakdeps = ["ForwardDiff"]

    [deps.NonlinearSolveQuasiNewton.extensions]
    NonlinearSolveQuasiNewtonForwardDiffExt = "ForwardDiff"

[[deps.NonlinearSolveSpectralMethods]]
deps = ["CommonSolve", "ConcreteStructs", "DiffEqBase", "LineSearch", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase"]
git-tree-sha1 = "3398222199e4b9ca0b5840907fb509f28f1a2fdc"
uuid = "26075421-4e9a-44e1-8bd1-420ed7ad02b2"
version = "1.2.0"
weakdeps = ["ForwardDiff"]

    [deps.NonlinearSolveSpectralMethods.extensions]
    NonlinearSolveSpectralMethodsForwardDiffExt = "ForwardDiff"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9216a80ff3682833ac4b733caa8c00390620ba5d"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.0+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "24be21541580495368c35a6ccef1454e7b5015be"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.11"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "5152abbdab6488d5eec6a01029ca6697dff4ec8f"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.23"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "6d38fea02d983051776a856b7df75b30cf9a3c1f"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.16"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "2e154f7d7e38db1af0a14ec751aba33360c3bef9"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.33.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "7cb9d10026d630ce2dd2a1fc6006a3d5041b34c0"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.14"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "341c75a6ba4fa155a2471f5609163df5e3184e7b"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.86.2"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMLStyleExt = "MLStyle"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLJacobianOperators]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "ConstructionBase", "DifferentiationInterface", "FastClosures", "LinearAlgebra", "SciMLBase", "SciMLOperators"]
git-tree-sha1 = "6e9d280334839fe405fdab2a1268f2969c9d3eeb"
uuid = "19f34311-ddf3-4b8b-af20-060888a46c0e"
version = "0.1.3"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "1c4b7f6c3e14e6de0af66e66b86d525cae10ecb4"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.13"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "566c4ed301ccb2a44cbd5a27da5f885e0ed1d5df"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "BracketingNonlinearSolve", "CommonSolve", "ConcreteStructs", "DifferentiationInterface", "FastClosures", "FiniteDiff", "ForwardDiff", "LineSearch", "LinearAlgebra", "MaybeInplace", "NonlinearSolveBase", "PrecompileTools", "Reexport", "SciMLBase", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "5e45414767cf97234f90a874b9a43cda876adb32"
uuid = "727e6d20-b764-4bd8-a329-72de5adea6c7"
version = "2.3.0"

    [deps.SimpleNonlinearSolve.extensions]
    SimpleNonlinearSolveChainRulesCoreExt = "ChainRulesCore"
    SimpleNonlinearSolveDiffEqBaseExt = "DiffEqBase"
    SimpleNonlinearSolveReverseDiffExt = "ReverseDiff"
    SimpleNonlinearSolveTrackerExt = "Tracker"

    [deps.SimpleNonlinearSolve.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseMatrixColorings]]
deps = ["ADTypes", "DocStringExtensions", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "76e9564f0de0d1d7a46095e758ae13ceba680cfb"
uuid = "0a514795-09f3-496d-8182-132a7b665d35"
version = "0.4.19"

    [deps.SparseMatrixColorings.extensions]
    SparseMatrixColoringsCliqueTreesExt = "CliqueTrees"
    SparseMatrixColoringsColorsExt = "Colors"

    [deps.SparseMatrixColorings.weakdeps]
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "f737d444cb0ad07e61b3c1bef8eb91203c321eff"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.2.0"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

    [deps.StaticArrayInterface.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "f35f6ab602df8413a50c4a25ca14de821e8605fb"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "PrettyTables", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "b6a641e38efa01355aa721246dd246e10c7dcd4d"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.40"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "18ad3613e129312fe67789a71720c3747e598a61"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.3"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f57facfd1be61c42321765d3551b3df50f7e09f6"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.28"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "068dfe202b0a05b8332f1e8e6b4080684b9c7700"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.47+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d5a767a3bb77135a99e433afe0eb14cd7f6914c3"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─8936bc61-2880-4cb8-acc5-17a6526d82c5
# ╠═478e4840-348f-11f0-38c2-d76b6a6ff32f
# ╠═3ccd44bb-e3e6-4318-ac7f-2047737c9842
# ╟─c05980ac-7862-4dee-8830-bd07d2ed63d9
# ╠═94a20e6c-d019-4e84-8f7c-832250cb503b
# ╟─af7d31c0-c829-478a-aff2-17c207c69a28
# ╠═d9226fcc-f6ba-41a9-ac51-90bc5b217597
# ╟─a3c7ee9b-d604-43a4-958a-3dea02ff2e4e
# ╠═f220ee45-ff4b-485f-bc25-f24598b7f4cf
# ╠═cd5eeabc-d26d-4789-a69a-8bd5de473770
# ╟─4e995b5b-427f-4c07-82a6-97ec356cbb10
# ╠═95be7081-5620-42e1-9f80-f3db9bed05b4
# ╟─a07a0555-c316-4f25-8e07-add4bdd859dd
# ╠═6f9762fa-c346-4e39-8afe-8a78113f7cb1
# ╟─833b5582-f1b1-44dc-beb9-60a21fbcc54c
# ╠═e377e101-692b-497e-995e-3baf1494f6a9
# ╟─2829f3ed-f4c8-4ce6-bb18-6cd6ee0ee43b
# ╟─f5711d64-d88d-46d2-9cac-5a74974d56a9
# ╟─d4f46f92-c225-46fd-aff9-c23424b86e02
# ╟─5a7d8ca9-fa14-49f4-8e1f-ed16b15a2a13
# ╠═eab4c524-5a81-4538-9070-fbb9822d1aea
# ╟─4aee11f0-6ab2-4088-8068-3e49158ad7c3
# ╠═230439df-c697-4b5e-acc3-b1ebc83be912
# ╠═c2b83e6c-a21a-466f-b587-bb9dc45c2909
# ╟─854d53fe-0f01-4383-88e7-11ba2588603e
# ╟─cf446947-b34d-47cb-b466-3b9b4cc64ab3
# ╠═fb75058c-c832-4e03-830a-5d058553bbd7
# ╟─57fce5e3-4bb6-4c45-bdeb-4c1ee7236d44
# ╟─782c9075-aa13-4ec8-b10c-4a9c9a3340e6
# ╠═b6173493-ce2e-4653-a89d-c6d4eaf32fd2
# ╟─d48632dd-57d0-4eec-a0cb-c715beb90c3c
# ╠═5a02d4e7-b86a-4531-a87e-f825199ab7da
# ╟─dc8dc412-5f89-4389-ae0d-91bcb8204827
# ╠═d25e8698-0625-435c-a78f-4b6b73d368c8
# ╟─214baa0c-c053-47ca-80d1-0139f3c8bd83
# ╠═c2c88bf6-891a-4025-91f4-915ab4a1af96
# ╟─efacf511-78cb-433b-ad3d-a8b08f483ec7
# ╠═f9f75345-7fe2-429a-b5fa-8d8f5876d379
# ╟─b2a6a10d-0397-4a2c-b41f-4b8e558ea720
# ╠═63b8f98c-2af7-4395-b2f6-22cff535e5c5
# ╟─a02be7d1-179f-43e4-9166-2580f6168cd5
# ╠═5cdf998e-ae4c-481b-b1a4-f74f6f3de85d
# ╟─fee67b8a-31f5-4fa8-8a5f-36ec7ea58e43
# ╠═7fdd9fd6-1f59-440a-bf44-46259ed78838
# ╟─de8d14bd-9ab9-4a7b-9071-eea87cf430e1
# ╠═3a0ec6de-035b-4b45-891c-97a3486217b7
# ╟─be9dfb11-5cc7-4e92-a47f-95d65313efbf
# ╠═7680f004-d427-42b3-9977-a7104b40adb8
# ╟─3e544dca-976a-459f-ba76-0d57ba11fedf
# ╠═ab737a60-f13d-4c61-9d78-742b1baef1af
# ╟─9e1133a3-90d3-4868-8313-7e66a1b489a3
# ╠═f10678f6-5e94-4142-b6a8-f96c2cfefdb8
# ╟─b0467ac4-1f11-421d-ac22-3022b7b5f8c8
# ╠═d9c607a1-dc4c-4a65-94bc-3255e965d970
# ╟─6780ee26-73f8-4aaa-8c7a-c8b2363c3dc6
# ╠═250b1666-173b-4308-873d-0ffc725f7f46
# ╠═66bcf94e-c3fd-4a1c-a5fa-d1e8f4cd261e
# ╠═08e76add-9ff1-4b33-8303-bbfacccf33de
# ╠═f2044bed-6181-4693-bf07-bd5270ee06e6
# ╟─0bf6cb63-6aab-454f-ba5c-2d7fd6e9f289
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
