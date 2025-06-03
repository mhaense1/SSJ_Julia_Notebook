### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 9bae8cdb-18bb-4c5b-8984-c27fd9417874
begin
	#Lets start with loading some Julia packages that will be useful below
	using Parameters, LinearAlgebra, SparseArrays
	using Setfield,ForwardDiff, BasicInterpolators
	using NonlinearSolve,Plots
	using QuantEcon: rouwenhorst
	using PlutoUI: TableOfContents
	using BenchmarkTools
end

# ╔═╡ 19ea0f40-4b3e-11ef-1820-4992f2ba942f
md"""

# SSJ Notebook II - General Equilibrium with and without DAGs

## by [Matthias Hänsel](https://mhaense1.github.io/)

This notebook serves as a sequel to my *[previous notebook](https://mhaense1.github.io/SSJ_Julia_Notebook/SSJ_notebook.html)* in which I demonstrated  how to solve a neoclassical Heterogenous Agents (HANC) business cycle model using the Sequence Space Jacobian (SSJ) linearization method proposed by  *[Auclert et al. (2021 - henceforth ABRS)](https://onlinelibrary.wiley.com/doi/full/10.3982/ECTA17434)*.

While the HANC model as in my original notebook is attractive for learning purposes due to its simplicity, it essentially abstracted from another challenge of the ABRS method, getting the General Equilibrium (GE) Jacobians.
ABRS propose a specific method for this, which is based on chaining partial equilibrium Jacobians along a Directed Acyclical Graph (DAG).

However, I personally found that just from reading the paper, it is not that easy to grap how that needs to be done in practice and while ABRS's comprehensive *[Python package](https://github.com/shade-econ/sequence-jacobian)* implements the DAG method very efficiently, its "black box" nature does not make it easy to figure it out by looking at their code.

Furthermore, once you actually understood it, it turns out that implementing the DAG method without a package automating everything can be somewhat tedious and error-prone (at least in my opinion). Thus, in this notebook I will not only implement the DAG method manually for a simple HANK model but also demonstrate **two alternative ways** to obtain the GE Jacobians:

1. **A Direct Method employing the HA Jacobians:**
    This idea is briefly mentioned in the ABRS paper and while inefficient, I find it conceptionally much simpler than the DAG method.


2. **A "Dynare-like" method:** 
    It is also possible to get Aggregate Jacobians by writing the aggregate parts of the model down in a way similar as one would e.g. in [Dynare](dynare.org). This idea was inspired by insights in [Bhandari et al. (BBEG)](https://www.nber.org/papers/w31744), who develop an interesting alterntive to the SSSJ method.

All methods are implemented relying solely on *automatic differentiation* through the `ForwardDiff.jl` package.

The example model for this notebook is the One-Asset HANK model also used in the original ABRS paper, which is briefly presented below. While still relatively simple, it has a more involved "macro" block due the presence of a Forward-Looking Phillips curve, endogenous labor supply and so on.
Additionally, ABRS have a [Python notebook](https://github.com/shade-econ/sequence-jacobian/blob/master/notebooks/hank.ipynb) also implementing the model so we can check the results here against the "original".
	
The notebook assumes some familiarity with Julia and features the following parts:

1. Description of the model
2. Solving/Calibrating the steady state 
3. Obtaining the Heterogeneous Agent Jacobian
4. Different ways to obtain the GE Jacobians

As previously, I tried to focus more on a *clear* and *simple* implementation instead one optimized for speed.
"""

# ╔═╡ a7d93065-0d0c-45ac-b497-79845fccf27a
TableOfContents()

# ╔═╡ 85614fee-9eea-4b9a-ac00-91847d8d7cce
md""" 
## The model

The description (mostly) follows the ABRS paper.

### Households

* There is a unit mass of ex-ante identical households who differ ex-post by their labor productivity $s$ (''skill'') and asset holdings $a$. 


* Skill $s$ follows a time-invariant discrete Markov Chain with $n_s$ states $\mathcal{S}:=\lbrace s_1,s_2,...,s_{n_s}\rbrace$ and exogenous transition probabilities $P(s',s)$. This introduces the income risk. $\pi_s$ denotes the stationary distribution of $P$. The average labor productivity $\int^1_0 s_{it} di$ is invariant and will be normalized to 1.


* Skill $s$ does not only determine an households income per hour worked, but also the amount of lump-sum taxes she has to pay and the amount of dividends she receives, both of which are distributed proportionally to $s$.


* A household can freely choose the amount of hours $n$ she works subject to an additively separable utility cost of working


* Savings- and labor choices $a$ and $n$ are the solution to the household's utility maximization problem, which is characterized by the following Bellman equation
```math
V_t(s,a) = \max_{a',n} \frac{c^{1-\sigma}-1}{1-\sigma} - \varphi\frac{n^{1+\nu}}{1+\nu} +\beta \sum_{s'\in \mathcal{S}}P(s',s)V_{t+1}(s',a') 
```
```math
\text{subject to}~~c+a' = (1+r_t)a + w_t n s - \underbrace{(d_t - \tau_t)}_{=\text{net transfer}}s ~~\text{and}~~a'\geq \bar{a}
```


* Above, $d_t$ and $\tau_t$ denote the aggregate level of dividends and taxes.


* We denote by $D$ the distribution of households, i.e. $D_t(s,a)$ is the mass of households who start the period with skill $s$ and assets $a$. 

### Firms

A competitive final goods firm assembles its output using a CES production function $Y_t = (\int^1_0 y_{jt}^{\frac{1}{\mu}} dj)^{\mu}$, giving rise to a standard CES-demand system for the continuum of intermediate goods. These intermediates, in turn,  are supplied by monopolists who produce using only labor s.t. $y_{jt}=Z_t l_{jt}$ and are subject to quadratic price adjustment costs a la Rotemberg (1982). $Z_t$ denotes aggregate productivity that may be time-varying. In a symmetric equilibrium, gross inflation $1+\pi_t = P_t/P_{t-1}$ can then be shown to evolve according to a New Keynesian Phillips curve of the form  
```math
\log(1+\pi_t) = \kappa\left(\frac{w_t}{Z_t}- \frac{1}{\mu} \right) + \mathbb{E}_t\frac{1}{1+r_{t+1}}\frac{Y_{t+1}}{Y_{t}}\log(1+\pi_{t+1}).
```
Here $\kappa$ is the parameter governing firms' price adjustment costs, $\mu$ the steady state markup and $Y_t$ aggregate output, which is equal to the monopolists' individual output by symmetry. The firms' profits are distributed lump-sum to households, thus $d_t = Y_t - w_tL_t$.

### Policy

The model features a government consisting of two branches, a **monetary** and a **fiscal** authority. The former sets the nominal interest rate $i_t$ according to a Taylor rule of the form
```math
i_t = r^* + \theta_\pi \pi_t + \epsilon^i_t
```
where $r^*$ is the economy's long run "natural rate" and $\epsilon^i_t$ a monetary policy shock. The real interest rate in period $t$ is determined by the previously set nominal rate and inflation so that $1+r_t = \frac{1+i_{t-1}}{1+\pi_t}$.

The fiscal authority, in turn, issues a constant amount of government bonds $B$ each period and collects the lump sum taxes $\tau_t$ already mentioned above. Since there are no other spending items, it chooses the tax rate to cover the its interest rate payments every period, i.e.
```math
\tau_t = r_t B~~.
```

### Market clearing

In an equilibrium, the following market clearing conditions will have to hold:

* Asset market:
```math
B = \int^1_0 a_{it} di 
```

* Labor market
```math
\frac{Y_t}{Z_t} = L_t = \int^1_0 s_{it} n_{it} di 
```

* Good market
```math
Y_t = \underbrace{\sum_{s\in \mathcal{S}}\int c(s,a) D(s,a)da}_{=\text{aggregate consumption}} - \underbrace{\frac{\mu}{\mu-1}\frac{1}{2\kappa}\left(\log(\pi_t)\right)^2Y_t}_{=\text{price adjustment costs}}
```

In this model, there are three markets that need to clear: Firstly, an asset market on which the sum of households' net savings need to equal the aggregate amount of bonds $B$ supplied by the government. Secondly a labor market on which the effective amount of labor supplied by the households' at the piece-rate real wage $w_t$ needs to equal the labor demanded by the firm. Finally, the goods market will have to clear, requiring aggregate consumption and real price adjustment costs to equal aggregate output.
As is well known, price adjustment costs don't matter in linearized models and thus the latter will not be of concern here.


### Calibration:

The calibration follows the example model in the ABRS paper and can be seen in the code block below.
The skill process follows an AR(1)-process in logs that will be discretized to 7 points using the Rouwenhorst method 
(the function implementing the Rouwenhorst method is off-the-shelf from *[QuantEcon.jl](https://github.com/QuantEcon/QuantEcon.jl)*).

"""

# ╔═╡ e42b4882-d822-43f7-8bec-4ebaffb5d17b
#Calibration of the model
@with_kw struct ModelParameters{T}

	#HH parameters
	β::T = 0.98 	#Household discount factor - will be changed by calibration
	φ::T = 0.8      #labor disutility - will be changed by calibration
	σ::T = 2.0   	#relative risk aversion
	ν::T = 2.0 		#inverse Frisch elasticity
    ρs::T = 0.966 	#persistence of HH income process
    σs::T = (1.0-ρs^2)^0.5 * 0.5 #variance for household income process

	#aggregates
    μ::T  = 1.2 	#steady state markup 20%
	κ::T  = 0.1     #Slope of NK Phillips curve
	B::T  = 5.6     #government bond supply
	θ_π::T = 1.5    #CB reaction to inflation
	Z_ss::T = 1.0   #steady state labor productivity

	#calibration target
	r_target::T = 0.005
	Y_target::T = 1.0
	
end

# ╔═╡ 7097a48b-8906-4197-8012-1b3673ab1d78
md"""
Two parameters will be calibrated internally below: The discount factor $\beta$ will be chosen to ensure bond market clearing at a target real interest rate of 0.5% and the labor disutility parameter $\varphi$ is set so that steady state effective labor supply (and thus output) equals 1.

The discretization of the model will work as in my *[previous notebook](https://mhaense1.github.io/SSJ_Julia_Notebook/SSJ_notebook.html)* notebook and thus I will not go into these details here. Below I define an additional structure to store some additional useful objects.
"""

# ╔═╡ d0385d4e-08ca-43b1-ae70-8da38175cb6e
function log10_grid(amin,amax, na)
	pivot = 0.25
	grid = Iterators.map(exp10, range(log10(amin + pivot),log10(amax + pivot), na)) .- pivot
	grid[1] = amin
	return grid
end

# ╔═╡ dbf4d282-2c04-4d9d-b1f1-2ad8f6e9098b
@with_kw struct NumericalParameters{F,I}

	#include instance of model parameters
	mp::ModelParameters{F} = ModelParameters()

	# specification for grids
	na::I 	= 500 	#no. of points on asset grids
	ns::I 	= 7 	#no. of points on skill grid
	amin::F = 0.0 	#borrowing limit
	amax::F = 150.0 #upper end grid

	#set grids and transition matrix for s process
	s_grid::Vector{F} 	= exp.(rouwenhorst(ns, mp.ρs, mp.σs).state_values)
	Ps::Matrix{F} 		= rouwenhorst(ns, mp.ρs, mp.σs).p
	a_grid::Vector{F} 	= log10_grid(amin,amax,na)

	#tolerance value for HH problem
	tolHH::F = 1e-9

	#size of Sequence Space Jacobians
	T::I = 300
end

# ╔═╡ 5f99a5dd-abab-4fd4-bb67-0ba7dd201977
md"""
## Solving the Household problem

For solving the household problem, we will again use the Endogenous Grid Method (EGM), which has to be modified to account for household's endogenous labor supply:

For given *next period* marginal value functions/consumption policy functions, we can still invert the Euler equation, but due to the endogenous labor supply choice, we don't immediately know how much wealth an household must have had to chose that consumption policy. However, we can link household's consumption and labor supply through the respective optimality condition

```math
\varphi n_{it}^\nu = w_t s_{it} c_{it}^{-\sigma}~~,

```
which requires the marginal disutility of working more to equal the marginal utility of the income the household earns from doing so. We can then implement the EGM backward step with the function below:
"""

# ╔═╡ 3708fb95-a65d-4a1f-bb5e-f4db9a8766b9
#compute expected marginal value function
function update_EVa(cpol::AbstractArray,r::T,np::NumericalParameters) where T<:Real
    return (1.0+r).*((cpol.^(-np.mp.σ)))*np.Ps'   
end

# ╔═╡ 78f02c49-becd-4060-9f2f-ac250f468110
md"""
The only slightly tricky thing is that for borrowing-constrained households, we can't find consumption through an inverted Euler equation. For these, we need to solve the above optimality condition for labor supply numerically. The function below (used in `EGM_update` above)  does that using fixed-point iteration:
"""

# ╔═╡ 60fa7af4-1bd9-4acb-bfe0-61801317b177
#solve consumption of constrained HHs via fixed point iteration
function solve_c_n_constrained(a_grid_c,c_guess,ws,r,T,np::NumericalParameters;
								max_iter = 100, tol = 1e-11)
	@unpack mp, amin = np
	@unpack σ,φ,ν = mp
	
	iter = 0 ; dist = 100.0
	while (iter < max_iter) & (dist > tol)

		#labor supply implied by consumption guess
		n_c = ((ws*(c_guess).^(-σ))./φ).^(1/(ν))

		#consumption implied by n_c (assuming constraint binding)
		c_implied = ws*n_c .+ T .+ (1+r)*a_grid_c .- amin

		dist = maximum(abs.(c_implied .- c_guess))

		#update
		c_guess = 0.25*c_implied .+ 0.75*c_guess
		iter += 1
	end

	if iter == max_iter
		@warn "Constrained labor supply didn't solve"
	end
	
	return c_guess

end

# ╔═╡ 7e4416f4-1211-45bc-9511-e17028af154b
function EGM_update(cPrime::AbstractArray,w,T_level,r,rPrime,np::NumericalParameters)

    @unpack mp,a_grid,s_grid,ns,na,amin = np
    @unpack β,σ,φ,ν = mp

	#get expected marginal utility
    EVa = update_EVa(cPrime,rPrime,np)

	#invert Euler equation
    MU_endog = (β.*EVa)

	#use labor supply optimality condition to back out n on endog grid
	n_endog = ((w*MU_endog.*s_grid')./φ).^(1/(ν))

	#consumption on endog. grid
	c_endog = MU_endog.^(-1/σ)

	#check consumption is positive
	@assert all(x -> x>0, c_endog)

    #vector of labor/transfer incomes for different types
    inc_endog   = w.*s_grid'.*n_endog 
    
    #calculate assets today consistent with savings choice
    a_endog = (c_endog .- inc_endog .- T_level*s_grid' .+ a_grid)./(1.0 + r)

    #below the points, borrowing constraints become binding
    constr = a_endog[1,:]

	#pre-allocate array
    c = zeros(eltype(n_endog),na,ns)

	#for interoperability with ForwardDiff:
    ag = a_grid .+ zeros(eltype(n_endog),1)
	#ensures that the grid has the same type as a_endog

    @views for si = 1:ns #loop over ind. producitivity state  

        #interpolate consumption, ignoring constraint for now
        itp_c = LinearInterpolator(a_endog[:,si],c_endog[:,si],NoBoundaries()) 
			
        c[:,si] = itp_c.(ag)

		#find constrained grid points
		constr_ind = (a_grid .< constr[si])

		#for constrained households: back out consumption using the labor supply optimality condition
		if any(constr_ind)
			c[constr_ind,si] = solve_c_n_constrained(a_grid[constr_ind],
											c[constr_ind,si],
											w*s_grid[si],r,T_level*s_grid[si],np)
		end
        
    end

	#labor supply
	n = ((w*((c).^(-σ)).*s_grid')./φ).^(1/(ν))

	#savings policy
	aPrime = (1+r)*a_grid .+ (w*n .+ T_level).*s_grid' .- c

	#impose borrowing limit exactly
	aPrime[aPrime .< constr'] .= amin
   
    return c,aPrime, n 

end

# ╔═╡ 3e960018-54bf-4331-9ac0-e130e2393778
md"""
Armed with the above functions, we can obtain steady state household policies by iterating over the `EGM_update` function for given $w$, $r$ and net transfers.

For convenience below, I also define a version of the function that starts with a hard-coded initial guess.

"""

# ╔═╡ dfb79d2c-d29c-4e27-b677-959b3d7b57f1
begin #begin codeblock

function solve_EGM_SS(cGuess::AbstractArray,wSS,
						rSS,T_level,np::NumericalParameters;  
						max_iter::Int = 3000, print::Bool = false)

	c0 = cGuess  
	
	local c1, a1, n1

	#iterate until convergence
	dist = 1.0; iter = 0
	while (dist>np.tolHH) & (iter < max_iter)
		
		c1, a1, n1 = EGM_update(c0,wSS,T_level,rSS,rSS,np)

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
	
	return c1, a1, n1
end

#helper function providing initial guess
function get_cguess(w,r,np)
	incs = w*np.s_grid 
 	return 0.1 .+ 0.1*((1+r)*np.a_grid .+ incs')
end

#additional method definition that calls get_cguess
function solve_EGM_SS(wSS,rSS,T_level,np::NumericalParameters; 
						max_iter::Int = 3000, print::Bool = false)

	c_guess = get_cguess(wSS,rSS,np)
	
	return solve_EGM_SS(c_guess,wSS,rSS,T_level,np, max_iter = max_iter, print = print)
end


end #end codeblock

# ╔═╡ b18c850d-1edd-4d30-b67f-5c6a21f2a7fc
md"""
Let's briefly test this for some values:
"""

# ╔═╡ 3dc4a1f0-7d85-42c9-a8e3-4de80f604b1b
solve_EGM_SS(1/1.2,0.01,0.01,NumericalParameters(),print = true)

# ╔═╡ cd3b00b6-cfc5-4096-8c63-601ae7cd214e
md"""
That worked, nice!

For getting the steady state distribution, I again use the Young (2010) histogram method. Since this was already discussed in the *[previous notebook](https://mhaense1.github.io/SSJ_Julia_Notebook/SSJ_notebook.html)*, I don't go into details and the functions that build the necessary transition matrix $\Lambda$ and solve for the stationary distribution $D$ are relegated to the hidden code fields below (they are called `build_Λ` and `inv_dist`, respectively).

"""

# ╔═╡ b1d47ec8-4958-449a-b9df-9f4cca062ed9
function build_Λ(a_choice::Matrix, np::NumericalParameters)

    @unpack  na, ns, a_grid, Ps = np

	#pre-allocate arrays
    weights_R = zeros(eltype(a_choice),na*ns,ns)
    weights_L = zeros(eltype(a_choice),na*ns,ns)
    IDX_col_R = zeros(Int64,na*ns,ns) #where agents "go"

    @views begin
        for ss = 1:ns
            for aa = 1:na

            #find closest smaller capital value on grid  
            al_idx = searchsortedlast(a_grid,a_choice[aa,ss])
    
            ss_shifter = (ss-1)*na #inner helper variable

                    #if clause checks for boundary cases
                    if al_idx == na   
					#if higher than highest grid point, I assign entire mass to maximum grid point	

                        for sss = 1:ns
                            weights_R[ss_shifter + aa,sss] += Ps[ss,sss]
                            IDX_col_R[ss_shifter + aa,sss] = al_idx + (sss-1)*na
                        end

                    elseif al_idx == 0 
						#if lower than smallest grid point, I assign entire mass to lowest grid point	

                        for sss = 1:ns
                            weights_L[ss_shifter + aa,sss] += Ps[ss,sss]
                            IDX_col_R[ss_shifter + aa,sss] = (sss-1)*na + 2
                        end

                    else    #regular case - using weights formula

                        wr = ((a_choice[aa,ss] - a_grid[al_idx]) / (a_grid[al_idx+1] 																	- a_grid[al_idx]))
                        lr = 1.0 - wr

                        for sss = 1:ns
                            weights_R[ss_shifter + aa,sss] += Ps[ss,sss]*wr
                            weights_L[ss_shifter + aa,sss] += Ps[ss,sss]*lr                         
                            IDX_col_R[ss_shifter + aa,sss] = (sss-1)*na + al_idx + 1
                        end

                    end

            end
        end

        IDX_from = repeat(1:(na*ns),outer=2*ns)
        weights = vcat(weights_R[:], weights_L[:])
        IDX_to = vcat(IDX_col_R[:],IDX_col_R[:] .- 1)
    end

    #return sparse transition matrix
    return sparse(IDX_from,IDX_to,weights,na*ns,na*ns)

end

# ╔═╡ 12e89139-f672-4265-bd5a-5639e3f96a5f
begin
	function inv_dist(Π::AbstractArray)
		#Π is a Stochastic Matrix
	    x = [1; (I - Π'[2:end,2:end]) \ Vector(Π'[2:end,1])]
	    return  x./sum(x) #normalize so that vector sums up to 1.
	end
end

# ╔═╡ 7ea684fb-e2a5-4935-b18e-f491b4c098b4
md"""

## Solving and Calibrating the Steady State

We are now ready to calibrate/solve the model's steady state. As mentioned above, I will choose $\beta$ so as to induce an SS interest rate of $r=0.005$ and $\varphi$ to achieve $L=Y=1$. The below function implements this:

"""

# ╔═╡ a40334ec-d57f-477e-97ad-9aa82b7ff46d
#ojective function for solver to calibrate β and φ
function β_φ_objective(β,φ,p,np)

		@unpack s_grid = np
		w_ss, r_ss, T_level_ss = p
	
		@set! np.mp.β = β
		@set! np.mp.φ = φ

		#solve HH problem
		c_ss, a_ss, n_ss = solve_EGM_SS(w_ss,r_ss,T_level_ss,np)

		#build transition matrix
		Λ =  build_Λ(a_ss, np)

		#get invariant distribution
		D = inv_dist(Λ)

		#aggregate capital stock implied by HH savings
		A_ss = sum(reshape(D,(np.na,np.ns)).*np.a_grid)
		L_ss = sum((reshape(D,(np.na,np.ns))).*(n_ss.*s_grid'))

	    dists = [(A_ss/np.mp.B) - 1, (L_ss/np.mp.Y_target) - 1]

		#return relative distance to target
		return dists
		
	end

# ╔═╡ 42e36f72-e02c-435f-8fa6-30309b3ee3e6
#Note: Since NumericalParameters is an immutable struct, I use @set!
	#rom the Setfield package to change its elements.
	
	function get_steady_state(np::NumericalParameters)

		#normalize avg. labor efficiency to 1
		s_dist = inv_dist(np.Ps)
		@set! np.s_grid = np.s_grid./sum(s_dist.*np.s_grid)

		#unpack objects
		@unpack mp, s_grid, na, ns = np
		@unpack μ, B, Y_target, r_target, Z_ss = mp

		r_ss = r_target
		Y_ss = Y_target ; L_ss = Y_ss/Z_ss

		#get steady state objects implied by parameters:
		w_ss = 1/μ #wage
		div_ss = Y_ss - w_ss*L_ss #dividend
		τ_level_ss = r_target*B  #tax rate
		T_level_ss = div_ss - τ_level_ss #net lump sum transfer

		#Solve for β and φ consistent with targets using Broyden's Method
		prob = NonlinearProblem((x,p) ->
					β_φ_objective(x[1],x[2],p,np),
					[np.mp.β,np.mp.φ],[w_ss,r_target,T_level_ss])
		sol = solve(prob,Broyden(alpha = 100.0))
		#Note: alpha = 100.0 ensures that the solvers does not take too big steps
		#at the beginning

		if sol.retcode != :Success
			@warn "Calibration didn't solve"
		end

		@set! np.mp.β = sol.u[1]
		@set! np.mp.φ = sol.u[2]

		#get remaining steady state objectives for correct β and φ
		c_ss, a_ss, n_ss = solve_EGM_SS(w_ss,r_target,T_level_ss,np)

		#get distribution as (na × ns matrix)
		Λ_ss = build_Λ(a_ss, np)
		D_ss = inv_dist(Λ_ss)
		D_ss = reshape(D_ss,(np.na,np.ns))

		#aggregate capital stock implied by HH savings
		A_ss = sum(D_ss.*np.a_grid)
		L_ss = sum(D_ss.*(n_ss.*s_grid'))
		N_ss = sum(D_ss.*n_ss) #aggregate hours
		C_ss = L_ss*Z_ss #aggregate consumption

		#check that Walras' Law holds
		println("Goods market clearing residual: ",abs(sum(D_ss.*c_ss) - C_ss))

		return (; L_ss, w_ss, Y_ss, r_ss, τ_level_ss, div_ss, T_level_ss, c_ss, 					a_ss, n_ss, Λ_ss, D_ss, A_ss, N_ss, C_ss ,np)

	end

# ╔═╡ 1e2773ef-0dd5-4c6e-ae9c-93001fd2e0ff
SS_objs = get_steady_state(NumericalParameters())

# ╔═╡ 54402a9f-56df-439f-98b8-64d15ad0705e
md"""
We can check our calibrated parameter values for $\beta$ as well as $\varphi$ and see that they correspond to those obtained by ABRS:
"""

# ╔═╡ 2f8230fd-a3b6-42c9-8ebc-fed90f15733e
SS_objs.np.mp

# ╔═╡ d9e401e9-b253-425f-9d98-340088757c2a
md"""

## Sequence Space Jacobians: Heterogeneous Agents (HA) Block

Having obtained the Steady State of our economy, we now wish to analyze the effects of a monetary policy shock to the central bank's nominal interest rate $i$.

To do that, let's briefly recap the SSJ method: It is based on the idea that an equilibrium in the sequence space can be expressed as a solution to a system of the form 
 ```math
F(X,Z) = 0
```
where $X$ denotes the time-paths of a vector of endogenous variables and $Z$ the time-paths of a vector of exogenous shocks. To get linearized Impulse Response Functions (IRFs), we can truncate the system $F$, compute its Jacobians with respect to $X$ and $Z$ and then obtain the IRFs as $dX = -F_X^{-1}F_Z dZ$ by the Implicit Function Theorem. 

Now, the problem is that in an heterogenous agent model, $F$ is potentially quite costly to evaluate, as within the system we need to keep track of household's policy function along the grids and the joint income- and wealth distribution. To overcome the issue, ABRS proposed the efficient "fake news"-algorithm to separately linearize a model "block" containing these complicated objects. 

Afterwards, one can then assemble $F_X$ and $F_Z$ around that "HA block". As said, this notebook deals with different ways to do so, but first, let us get the HA Jacobians using the ABRS "fake news" algorithm. Explaining in detail how it works would require quite a bit of text, so read the paper for that. As in the previous notebook, I'll just show you the implementation.

First define a function to get the object denoted as $\mathcal{E}$ in their paper:

"""

# ╔═╡ acf9f7a7-300f-423d-8220-6014327ec39b
function get_𝓔(Λ,y_ss,np::NumericalParameters)

	𝓔 = zeros(length(y_ss),np.T+1)
	𝓔[:,1] = y_ss[:]

	for tt = 2:(np.T+1)
		@views 𝓔[:,tt] = Λ*𝓔[:,tt-1]
	end

	return 𝓔
	
end

# ╔═╡ 9c53d4a7-526d-4065-88b5-0ef5e04f9b62
md"""
Next, the below gets us the objects $\mathcal{D}$ and $\mathcal{Y}$ for which we need to conduct the backwards iteration. This works as in the previous notebook, except that we have more inputs and outputs of the HA block.
"""

# ╔═╡ 575be7fa-ae2f-4fac-b7b7-c6ae6a1fec93
#function that conducts the backwards iterations, taking as inputs the SS and perturbations of w and r and the level of net transfers (dividends minus lump sum taxes)

	function backward_objective(dx_w,dx_r,dx_T,c_ss,w_ss,r_ss,T_level_ss,
									D_ss,np::NumericalParameters)

		#Note: dx_w and dx_r are the perturbations to the wage and interest rate 
		#T periods in advance
		
		@unpack T, s_grid = np

		type_ind = dx_w+dx_r+dx_T #get type of perturbance term to initialize containers

		#initialize containers
		A_terms = zeros(eltype(type_ind),T)
		L_terms = zeros(eltype(type_ind),T)
		D_terms = zeros(eltype(type_ind),length(D_ss),T)

		#get values for re-centering 
		#(cf. Appendix C of Auclert et al. (2021))
		c_nc = EGM_update(c_ss,w_ss,T_level_ss,r_ss,r_ss,np)[1]
		
		c_rc = c_ss

		#backward iteration
		for tt in Iterators.reverse(1:T)

			if tt==T

				c_t, aPrime_t, n_t = EGM_update(c_rc,w_ss + dx_w, T_level_ss + dx_T,
												r_ss + dx_r,r_ss,np)

			elseif tt == (T-1)

				c_t, aPrime_t, n_t = EGM_update(c_rc,w_ss,T_level_ss,r_ss,r_ss + dx_r,np)

			else

				c_t, aPrime_t, n_t = EGM_update(c_rc,w_ss,T_level_ss,r_ss,r_ss,np)

			end

			 Λ = build_Λ(aPrime_t,np)

			@views begin #@views reduces the number of allocations
			 A_terms[tt] = (aPrime_t[:])'*D_ss[:]
			 L_terms[tt] = ((n_t.*s_grid')[:])'*D_ss[:]
			 D_terms[:,tt] = Λ'*D_ss[:]
			end

			#do re-centering
			c_rc = c_ss .+ (c_t .- c_nc)

		end

		return vcat(A_terms,L_terms,D_terms[:])
	end

# ╔═╡ fed0b8e1-fe6c-4c17-8c03-bc819ac7c45d
#function that returns HA SSJ for given parameters and SS inputs

function get_𝒴_𝒟(c_ss,w_ss,r_ss,T_level_ss,D_ss,np::NumericalParameters)

	@unpack T, na, ns = np

	#objective function to be differentiated
	obj_fun(x) = backward_objective(x[1],x[2],x[3],c_ss,w_ss,r_ss,T_level_ss,D_ss,
										np::NumericalParameters)

	#automatic differentiation using ForwardDiff
	derivs = ForwardDiff.jacobian(x -> obj_fun(x),zeros(3))

	#note given way the outputs of obj_fun are saved, need to flip arrays
	#(done by reverse())
	𝒴 = Matrix{Vector{Float64}}(undef,3,2)
	𝒟 = Vector{Matrix{Float64}}(undef,3)
	
	for ii = 1:3

		𝒴[ii,1] = reverse(derivs[1:T,ii]) 
		𝒴[ii,2] = reverse(derivs[T+1:2*T,ii])  
		𝒟[ii] = reverse(reshape(derivs[(2*T+1):end,ii],(na*ns,T)),dims=2)

	end

	return 𝒴, 𝒟
	
end


# ╔═╡ ca2de9d9-879f-46ac-a02e-b5af6ad20f6f
md"""
Finally, a function that assembles the "fake news" matrix $\mathcal{F}$ and the  HA block Jacobian $\mathcal{J}$:

"""

# ╔═╡ 0303d80f-a072-4a51-bfaf-d37f721fe82f
function build_𝒥(𝒴,𝓓,𝓔,np::NumericalParameters)

	#construct fake news matrix 𝓕
	
	𝓕 = zeros(np.T,np.T)

	𝓕[1,:] = 𝒴 
	𝓕[2:end,:] .= 𝓔[:,1:(np.T-1)]'*𝓓

	#assemble the Sequence Space Sacobian 𝒥

	𝒥 = zeros(np.T,np.T)
	𝒥[:,1] .= 𝓕[:,1]
	𝒥[1,2:end] .= 𝓕[1,2:end]

	for tt = 2:np.T
		@views 𝒥[2:end,tt] .= 𝓕[2:end,tt] .+ 𝒥[1:(end-1),tt-1]
	end
	
	return 𝒥, 𝓕

end

# ╔═╡ e9772d25-e08b-4888-833c-1a3e79518b22
md"""
With this, we can now get our HA Jacobians. We have 3 blocks inputs ($r$, $w$ and the net transfers $d-τ$) and two outputs (aggregate labor supply $\mathcal{N}=\int^1_0 s_{i}n_{i}di$ and net savings $\mathcal{A}=\int^1_0 a_i di$), so need to construct 6 Sequence Space Jacobians in total. The code below does that using the functions defined above.
"""

# ╔═╡ dac31547-cebf-4c1b-be47-5253b4b86abc
begin

#put SSJs together
	
@unpack c_ss,w_ss,r_ss,np,D_ss,a_ss,n_ss,T_level_ss,np = SS_objs

	𝒴s, 𝓓s = get_𝒴_𝒟(c_ss,w_ss,r_ss,T_level_ss,D_ss,np)

	#container for Jacobians
	Js = Matrix{Matrix{Float64}}(undef,3,2)
	
	for oo = 1:2 #loop over HA block outputs
		if oo == 1
			x_ss = a_ss
		else
			x_ss = n_ss.*np.s_grid'
		end
		
		𝓔 = get_𝓔(SS_objs.Λ_ss,x_ss,SS_objs.np)
		
		for ii = 1:3 #loop over HA block inputs
			Js[ii,oo] = build_𝒥(𝒴s[ii,oo],𝓓s[ii],𝓔,np::NumericalParameters)[1]	
		end
	end
	
end

# ╔═╡ 6da15895-81a6-434e-9002-522a3b3348a1
md"""

## GE Jacobians: The DAG approach

Having obtained the SSJ's of the HA block, we now want to obtain the aggregate Jacobians $F_X$ and $F_Z$. For the former, it is often desirable to state it in terms of the minimum number of variables necessary: $F_X$ will have dimension $n_x \times T$, where $T$ is the truncation horizon and $n_x$ the number of variables in $X$. If $n_x$ becomes high, $F_X$ can become quite large and in turn computationally costly to invert.

To get $F_X$ and $F_Z$, can we follow the approach outlined in ABRS, which is to split the model into different "blocks" and then chaining their derivatives along a Directed Acyclical Graph (DAG). For the model used here, the ABRS paper proposes a DAG as displayed below:

![DAG graph](https://github.com/mhaense1/SSJ_Julia_Notebook/blob/main/ABRS_NK_DAG.png?raw=true)

"""

# ╔═╡ 7a2bba2b-7084-4979-82c8-6cf3e5daec3d
md"""

The aggregate representation is terms of just three aggregate variables, output $Y$, the real wage $w$ and inflation $π$ and a corresponding number of GE conditions, which consist of the NK Phillips curve as well as labor- and asset market clearing conditions as stated above. However, these variables imply others which enter the different blocks that in turn provide additional inputs for more blocks and so on.

Either the way, the below function implements obtains the Aggregate Jacobians by chaining along the DAG "manually". Since all the blocks except the HA one have straightforward analytical derivatives, I just directly enter the analytical ones but one could equivalently obtain them by using some differentiation packages.

For simplicity, I will assume all exogenous variables except the monetary policy shock (represented by $r^*$ in the ABRS DAG) fixed here. The implementation of further shocks is left as exercise to the reader (*I always wanted to write that phrase somewhere*).

Furthermore, I disregard terms w.r.t. the price adjustment costs since these turn out to be 0 anyway (feel free to check for yourself).

"""

# ╔═╡ fb08ff71-d0f9-4065-99ec-e4a1f6479072
#function to get GE jacobians
function get_GE_jacob_DAG(Js,SS_objs)

	#unpack parameters and SS objects
	@unpack np, w_ss, Y_ss, L_ss, r_ss = SS_objs
	@unpack mp, T = np
	@unpack B, μ, θ_π, κ, Z_ss = mp

	#TxT sparse identity ad 0 matrices
	I_sp = sparse(I,T,T) ; sp0 = spzeros(T,T)

	#### relevant derivatives firm block
	#inputs: Y, w ; outputs: d, L 
	dL_dY = 1/Z_ss
	dL_dw = 0.0
	dd_dY = 1 - w_ss*dL_dY
	dd_dw = -L_ss 

	#### relevant derivatives mon pol block
	#inputs: π, RShock ; output: r
	
	#derivatives of real rate w.r.t. inflation
	#since  r = (1 + r^* + θ π_ + ϵ_i)/(1+π) - 1 , both current and future π shows up
	dr_dπ = spdiagm(-1 => repeat([θ_π],T-1), 0 => repeat([-(1+r_ss)],T))
	dr_dRShock = spdiagm(-1 => repeat([1],T-1)) 

	#### relevant derivatives fisc block
	#inputs: r ; outputs: τ
	dτ_dr = B

	#### derivatives of targets

	## Target 1: NK Philipps curve (κ(w-1/μ) + 1/(1+r)π(+1) - π = 0)
	#inputs: π, w, Y, r
	dT1_dw = κ
	dT1_π = spdiagm(0 => repeat([-1],T), 1 => repeat([1/(1+r_ss)],T-1))
	dT1_Y = 0.0 ; dT1_r = 0.0

	##Labor market clearing (L - 𝒩 = 0)
	#inputs: L, 𝒩
	dT2_dL = 1.0
	dT2_d𝒩 = -1.0

	##Asset market clearing (A - B = 0 )
	#inputs: A
	dT3_dA = 1.0

	#assemble aggregate derivatives

	#for Target 1
	F_π_T1 = dT1_π ;
	F_π_T2 = dT2_d𝒩*(Js[2,2] .- Js[3,2]*dτ_dr)*dr_dπ ;
	F_π_T3 = dT3_dA*(Js[2,1] .- Js[3,1]*dτ_dr)*dr_dπ ;

	F_RShock_T1 = sp0

	#for Target 2
	F_Y_T1 =  sp0
	F_Y_T2 =  dT2_d𝒩*Js[3,2]*dd_dY .+ I_sp*dT2_dL*dL_dY
	F_Y_T3 =  dT3_dA*Js[3,1]*dd_dY

	F_RShock_T2 = dT2_d𝒩*Js[2,2]*dr_dRShock .+
					dT2_d𝒩*Js[3,2]*(-dτ_dr)*dr_dRShock

	#for Target 3
	F_w_T1 = (dT1_dw*I_sp)
	F_w_T2 = dT2_d𝒩*(Js[1,2] .+ dd_dw*Js[3,2])
	F_w_T3 = dT3_dA*(Js[1,1] .+ dd_dw*Js[3,1])

	F_RShock_T3 = dT3_dA*Js[2,1]*dr_dRShock .+
					dT3_dA*Js[3,1]*(-dτ_dr)*dr_dRShock

	#put the above together
	F_x = [F_π_T1 F_Y_T1 F_w_T1;
			F_π_T2 F_Y_T2 F_w_T2;
			F_π_T3 F_Y_T3 F_w_T3]

	F_RShock = [F_RShock_T1; F_RShock_T2; F_RShock_T3]

	return F_x, F_RShock

end

# ╔═╡ 3e4d39b1-34b2-4a63-a081-49db00e00866
md"""
While implementing the DAG method "manually" as above is perfectly doable for this not-that-complicated model, the above function looks somewhat tedious and I found it to be quite easy to make mistakes (try implementing if from scratch yourself). Naturally, a richer model would exacerbate this.

Nevertheless, we can now get the aggregate dynamics in response to a monetary shock. The code below obtains the model IRFs of inflation to a 25 basis point nominal rate disturbance that follows an AR(1)-process with different levels of persistence, ranging from 0.2 to 0.9.
"""

# ╔═╡ f7537f35-7549-4cdc-994d-616394084d3c
begin

	
#get F_x and F_R
F_x1, F_RShock1 = get_GE_jacob_DAG(Js,SS_objs)

#get vectors of monetary policy shocks with different persistence
dRShock = -0.0025.*([0.2,0.4,0.6,0.8,0.9]'.^(0:299))

#get aggregate variables
AggVars_DAG = (-(F_x1\Matrix(F_RShock1))*dRShock)

#extract inlfation response
π_dag = AggVars_DAG[1:300,:]

#plot inflation response
labels = ["ρ=0.2" "ρ=0.4" "ρ=0.6" "ρ=0.8" "ρ=0.9"]
plot(1:20,10_000*π_dag[1:20,:],  labels = labels, linewidth = 3.0, ylabel = "Basis Points", legend = :topright,title = "Inflation Response (DAG method)")
end

# ╔═╡ c23a83c1-4009-4378-b229-27b56cf4f640
md"""
Reassuringly, the responses look as in the [ABRS notebook](https://github.com/shade-econ/sequence-jacobian/blob/master/notebooks/hank.ipynb).

## GE Jacobians: Direct Method with HA Jacobians


While this is certainly a matter of taste, I dislike implementing the DAG method manually for models that are a bit more complicated, mainly because setting up the DAG structure, differentiating all the different blocks and chaining them starts feeling tedious at some point. 

Thus, I have been exploring different approaches to getting the aggregate Jacobians, which I'd like to share with the hope that others may find them useful as well.

I'll start with an alternative Method, which I'd call "Direct Methodd with HA Jacobian" and which is actually briefly outlined in the ABRS paper at the end of Section 4.1. While ABRS state it to be "less accurate", we will see that this concern is eliminated by the use of an AutoDiff package such as `ForwardDiff.jl`.

The idea is to set up the aggregate system of equation $F$ non-linearily but replace the non-linear HA-block with an linearized approaximation based on the HA SSJs we obtained using the "fake news" method. For the model analyzed here, this would look as follows:

"""

# ╔═╡ 90c792f4-eef4-4900-813d-b6dc29e8389c
function direct_HAjacob_objective(π_path,Y_path,w_path,RShock_path,Js,SS_objs)

		@unpack np,r_ss,T_level_ss, L_ss,Y_ss, w_ss = SS_objs
		@unpack mp = np
		@unpack θ_π, B, μ, κ, Z_ss = mp

		#implied path by labor productivity
		L_path = Y_path./Z_ss

		#implied nominal rates
		i_path = r_ss .+ θ_π*π_path .+ RShock_path

		#implied real rates - first period nominal rate is given by r_ss
		r_path =  (1 .+ vcat(r_ss,i_path[1:end-1]))./(1 .+ π_path) .- 1

		#paths for or r_{t+1}, π_{t+1} and Y_{t+1} - assumes economy back in SS after time T
		rp_path = vcat(r_path[2:end],r_ss) ; πp_path = vcat(π_path[2:end],0.0) 
		Yp_path = vcat(Y_path[2:end],Y_ss)

		#implied path of dividends
		div_path = Y_path .- w_path.*L_path

		#implied path of taxes
		τ_path = r_path*B

		#implied path of net transfers
		T_level_path = div_path .- τ_path 

		#labor supply path implied by linearized HA block
		𝒩_path = L_ss .+ Js[1,2]*(w_path .- w_ss) .+ Js[2,2]*(r_path .- r_ss) .+
							Js[3,2]*(T_level_path .- T_level_ss)
	
		#Asset demand path implied by linearized HA block
		A_path = B .+ Js[1,1]*(w_path .- w_ss) .+ Js[2,1]*(r_path .- r_ss) .+
							Js[3,1]*(T_level_path .- T_level_ss)

		#residual NK Philipps curve
		resid_1 = κ*(w_path./Z_ss .- 1/μ) .+ 
					(1.0./(1 .+ rp_path)).*(Yp_path./Y_path).*log.(1 .+ πp_path) .- 
					log.(1 .+ π_path) 

		#residual labor market clearing condition
		resid_2 = L_path .- 𝒩_path

		#residual assset market clearing condition
		resid_3 = A_path .- B

		return vcat(resid_1,resid_2,resid_3)

	end

# ╔═╡ 06413e4c-9b9e-4360-9222-3449bb751418
md"""

Again, we will construct the GE Jacobian in terms of same three aggregate variables that fully characterize the aggregate dynamics of our model economy: The above function takes as inputs time-paths of these variables as well as the monetary policy shock and returns the residuals of the "target" equations along the time path.

What I like about this approach is that you don't need to think that much about how to set up the DAG and you don't have to differentiate and chain a lot of small blocks separately. And it gives us the same results:
"""

# ╔═╡ 12e86c4c-dbae-4095-9c79-293ba38df5dc
#function that returns the Jacobians of F using the direct method
	function get_jacobs_direct(Js,SS_objs)

		@unpack np,r_ss,w_ss,Y_ss = SS_objs
		@unpack T = np

		#check that system of equationsfulilled
		check = direct_HAjacob_objective(zeros(T),ones(T),
							w_ss*ones(T),zeros(T),Js,SS_objs)

		#check that equations hold in SS
		if maximum(abs.(check)) > 1e-8
			@warn "large F residuals"
		end

		#differentiate system w.r.t endogenous variables
		F_x = ForwardDiff.jacobian( x -> 
				direct_HAjacob_objective(x[1:T],x[T+1:2*np.T],
				x[2*T+1:3*T],zeros(T),Js,SS_objs),
				vcat(zeros(T),ones(T),w_ss*ones(T)))

		#differentiate system w.r.t shocks
		F_RShock = ForwardDiff.jacobian( x -> 
					direct_HAjacob_objective(zeros(T),ones(T),
					w_ss*ones(T),x,Js,SS_objs),zeros(T))


		return F_x, F_RShock

	end

# ╔═╡ a79aebd9-44b4-4267-87bc-62b8c8723766
F_x2, F_RShock2 = get_jacobs_direct(Js,SS_objs)

# ╔═╡ e1ec7322-f8fc-428f-85db-04ccbabf023f
md"""
Indeed, we see that the Jacobians obtained using the DAG method and the Direct Method with HA Jacobian are the same (up to some small numerical error). Let's also plot the inflation response as visual demonstration.

"""

# ╔═╡ 24bdcddd-af1f-430c-bed5-3cda6536f4a1
Js[1,1]

# ╔═╡ ffc02cf0-d10f-4dcb-a33a-9705ac6ccb12
begin

println("Maximum difference for F_x: ",maximum(abs.(F_x1 .- F_x2)))
println("Maximum difference for F_RShock: ",maximum(abs.(F_RShock1 .- F_RShock2)))

#get aggregate variables
AggVars_direct = (-(F_x2\Matrix(F_RShock2))*dRShock)

#extract inlfation response
π_direct = AggVars_direct[1:300,:]

#plot inflation response
plot(1:20,10_000*π_dag[1:20,:],  labels = labels, linewidth = 3.0, ylabel = "Basis Points", legend = :topright,title = "Inflation Response (Direct method)")

end

# ╔═╡ 3ca6f293-b683-4c70-b9eb-93fff236b786
md"""
Given that it gives us what we want and seems easy, why not always use this method?

The issue is that the computational cost of getting the Jacobians of $F$ is much larger with the direct method, even if we use the HA Jacobian: In terms of computation time, for my implementations the direct method takes more than 50-100 times as long as the DAG method:

(Note that computation times displayed in the uploaded notebook differ from those I obtained on my laptop, as the code is run on a github server.)
"""

# ╔═╡ 404e75c6-77d3-48df-9376-4eaae9cc9827
@btime get_jacobs_direct($Js,$SS_objs)

# ╔═╡ 1a52ce21-ada0-4dea-b484-e6c4393328d4
@btime get_GE_jacob_DAG($Js,$SS_objs)

# ╔═╡ 6555e425-097c-45d3-9fc2-6c579ca099d5
md"""
That time difference shouldn't matter much if you are just interested in solving a model a few times for a given parameterization: After all, approx. 1 second for the direct method is not a long time. However, it will start to matter if you need to compute the Jacobians many times for estimating the model.

Thus, I think this "direct method" may be useful because it easy to implement and a) often models are just calibrated and not estimated anyway and b) even if that is the case, having a simple-to-implement additional method may be useful to check that a manually implemented DAG method gives the right results.
"""

# ╔═╡ 62e7a938-89b2-4123-b431-1f4cd98b3da8
md"""

## A "Dynare-like" method

Finally, reading the recent paper by [Bhandari et al. (2023)](https://www.nber.org/papers/w31744) features some analytical results suggesting yet another way to get the Aggregate Jacobians of $F$. I will use a somewhat different formulation to aid exposition, but that is where I got the inspiration from.

Instead of considering a big system $F$, they represent the model as a set of equations that need to hold in every period, as state space approaches implemented by DSGE modelling packages like *[Dynare](dynare.org)* typically do. In my own notation, we formulate the model as a system of equations $G$ that needs to hold every period, i.e. a perfect foresight equilibrium is defined as sequences $X$ so that
```math
G(\tilde{X}_t,\tilde{Z}_t,x_t)=0~~\text{with}~~x_t = H(\lbrace X_t\rbrace_{t=0}^\infty,\lbrace Z_t\rbrace_{t=0}^\infty)~~\forall t
```
for exogenously given $Z$, where $\tilde{X_t}=[X_{t-1},X_t,X_{t+1}]$ (and the same for $Z$) and $x_t$ denotes a set of variables that are the outcomes of the model's HA block, which I denote by $H$. In the model studied here, $x_t$ would be asset demand and labor supply.

Differentiating the LHS w.r.t $\tilde{X_t}$ , we have
```math
G_{\tilde{X}} + G_x\mathcal{J}_{t,\tilde{X}}~~\forall~t
```
and a similar expression if differentiating w.r.t. $Z$. I am obviously abusing notation here.

Anyway, the point is that we can construct the Aggregate Jacobians $F_X$ and $F_Z$ by writing down the aggregate model part $G$ in a Dynare-like fashion. The below code Julia function displays how the function $G$ can look like if we set it up for *all* model variables (not just $π$, $w$ and $Y$):

"""

# ╔═╡ 17da4c85-a1c8-4204-91df-001a17bd07c4
function G_fun(Xlag,X,XPrime,het_out,RShock,np::NumericalParameters)

	@unpack mp = np
	@unpack μ, κ, B, θ_π, Z_ss, r_target = mp

	#unpack aggregate variables
	wp, rp, T_levelp, divp, ip, Yp , τp , πip = XPrime
	w, r, T_level, div, i, Y , τ, πi = X
	w_, r_, T_level_, div_, i_, Y_ , τ_, πi_ = Xlag 

	#unpack HA block outputs
	L, A = het_out

	#helper variable to get correct eltype for container below
	ind_var = Yp + Y + Y_ + L + RShock

	#container for equation residuals
	eq_out = zeros(eltype(ind_var),length(Xlag)) 

	#NK Philips curve
	eq_out[1] += κ*(w/Z_ss - (1/μ)) + 1/(1+rp)*(Yp/Y)*log(1+πip) - log(1+πi)
	
	#Asset market clearing
	eq_out[2] += A - B

	#labor market clearing
	eq_out[3] += Y - Z_ss*L

	#dividendeds
	eq_out[4] += div - Y*(1-w/Z_ss)

	#taxes
	eq_out[5] += τ - r*B

	#net Transfers
	eq_out[6] += T_level - (div - τ)

	#nominal rate
	eq_out[7] += i - (r_target +  θ_π*πi + RShock)

	#dynamics of real interest rates
	eq_out[8] += r - ((1+i_)/(1+πi) - 1)

	return eq_out

end

# ╔═╡ 0610663b-a8a9-4182-86eb-98093e77d35b
md"""
The function takes as inputs current, lagged and (expected) next period values of the endogenous variables and shocks as well as the outputs of the HA block, denoted as `het_out` in the code.

As for State Space approaches, we need to ensure that the timing of the model is correct and the set of equation sufficient to fully characterize the equilibrium. Otherwise you'll get an error or weird results.

Now, to get the aggregate Jacobians of the model, we just need to differentiate `G_fun` with respect to the relevant inputs, set up some $(n_x\cdot T)\times (n_x\cdot T)$ matrices that have these derivatives at the right places, multiply or add them to the HA Block Jacobian $\mathcal{J}$ and we are done. It might require a bit of thinking to really understand how to do that correctly, in the function below I tried to do it in the simplest way possible.
"""

# ╔═╡ 2e128a66-9a3b-460a-9641-2bb8cba9c6b3
function get_jacobs_G(Js,SS_objs)

	np = SS_objs.np

	#define vectors of steady state variables
	Xss = [SS_objs.w_ss,SS_objs.r_ss,SS_objs.T_level_ss,SS_objs.div_ss,
			SS_objs.r_ss,SS_objs.Y_ss,SS_objs.τ_level_ss,0.0]
	het_out_ss = [SS_objs.L_ss,SS_objs.A_ss]

	#check that equations hold in SS
	check = G_fun(Xss,Xss,Xss,het_out_ss,0.0,np)
	if maximum(abs.(check)) > 1e-8
		@warn "large F residuals"
	end

	#compute derivatives derivatives
	dG_X = ForwardDiff.jacobian(x -> 
				G_fun(Xss,x,Xss,het_out_ss,0.0,np),Xss)
	
	dG_Xlag = ForwardDiff.jacobian(x -> 
				G_fun(x,Xss,Xss,het_out_ss,0.0,np),Xss)
	
	dG_XPrime = ForwardDiff.jacobian(x -> 
				G_fun(Xss,Xss,x,het_out_ss,0.0,np),Xss)
	
	dG_het = ForwardDiff.jacobian(x -> 
				G_fun(Xss,Xss,Xss,x,0.0,np),het_out_ss)
	
	dG_RShock = ForwardDiff.derivative(x -> 
				G_fun(Xss,Xss,Xss,het_out_ss,x,np),0.0)

	#define a selection matrix for HA block inputs
	P = zeros(3,8) ; P[1,1] = 1.0 ; P[2,2] = 1.0 ; P[3,3] = 1.0
	#Note the ordering: w is first HA block input and also first element of X etc.

	#define some helper matrices
	Im = sparse(I,np.T,np.T)
	Im_ = spdiagm(-1=>ones(np.T-1))
	Imp = spdiagm(1=>ones(np.T-1))

	#get J into suitable shape (again, note ordering of variables)
	J_reshaped = [hcat(Js[:,2]...) ; hcat(Js[:,1]...)]
	
	#assemble Aggregate Jacobians
	F_x = kron(dG_het,Im)*J_reshaped*kron(P,Im) .+ 
			kron(dG_X,Im) .+ kron(dG_XPrime,Imp) .+ kron(dG_Xlag,Im_)
	F_RShock = kron(dG_RShock,Im)

	test_M1 = kron(dG_X,Im) .+ kron(dG_XPrime,Imp) .+ kron(dG_Xlag,Im_)
	test_M2 = kron(dG_het,Im)
	test_M3 = J_reshaped
	test_M4 = kron(P,Im) 
	
	return F_x, F_RShock

end

# ╔═╡ 09f07dd5-a950-44f7-8a20-6070a8fa45b3
md"""
Since $F_X$ and $F_Z$ are different here as they relate to bigger number of variables, let us instead test whether we get the same results as before:
"""

# ╔═╡ 48e1498e-2d8b-4051-b90c-799ab37c3338
begin 

#get Aggregate Jacobians
F_x3, F_RShock3 = get_jacobs_G(Js,SS_objs)

#obtain impulse response
AggVars_dyn = -F_x3\Matrix(F_RShock3)*dRShock

#extract inflation response
π_dyn = AggVars_dyn[2101:2400,:]

println("Max. abs. difference compared to DAG: ", maximum(abs.(π_dyn .- π_dag)))

plot(1:20,10_000*π_dyn[1:20,:],  labels = labels, linewidth = 3.0, ylabel = "Basis Points", legend = :topright,title = "Inflation Response (G method)")
	

end

# ╔═╡ 6f160a63-8fec-4f8b-a2ea-0dc691e30d30
md"""
As the plot above confirms, the model responses to the nominal rate shocks are the same as for the DAG method (up to some Floating Point Error). And in contrast to the Direct method, getting $F_X$ and $F_Z$ is rather fast:
"""

# ╔═╡ fb7958ac-014b-40cb-9020-7d6f645a7f87
@btime get_jacobs_G($Js,$SS_objs);

# ╔═╡ b2897cf6-720a-45b5-860d-85044cdd7f28
md"""
Nevertheless, while the above worked and conveniently returns the Jacobians and time-paths for *all* model variables at the same time, that is not always desirable: Effectively, that gave us a bigger $F_X$-Matrix and inverting big matrices is costly. Here the difference is not so big, but if you have, say, 20 aggregate variables and a longer truncation horizon (say $T=500$), it will start to matter.
"""

# ╔═╡ 53188374-85e0-41f5-aa3b-a89a339cd974
@btime -($F_x3)\Matrix(($F_RShock3));

# ╔═╡ 7de8f598-dd30-428a-afb6-4cc372e1c4f7
@btime -($F_x1)\Matrix(($F_RShock1));

# ╔═╡ 9790e1b7-cb54-4086-ba92-4006667359b0
md"""
So, can we implement the above method for a reduced number of aggregate variables, say, again $π$, $w$ and $Y$? Indeed, I will demonstrate that below.
"""

# ╔═╡ 331dfd8e-c4ad-4c6d-89c9-691eb5a16f0c
md"""
## "Dynare"-like method: Alternative Formulation

Compared to the case above, an issue is that the minimum set of aggregate variables necessary to characterize the dynamics of the economy may not contain all the inputs of the HA block. 

For example, the households' wages and assset returns may both be a function of the aggregate capital stock. Or, for this model, if we again choose $π$, $w$ and $Y$ as aggregate variables, the household does not care about inflation per se but rather the real return it induces in conjunction with the nominal rate set by the central bank.

If that is the case, compared to above, we need to write an additional function that takes the aggregate $X$ and shocks $Z$ as inputs and returns the variables the housheolds actually care about (i.e. the inputs of the HA Block). The function below does just that:

"""

# ╔═╡ f0cdbfc2-6216-491b-a874-0313e20558b4
#returns takes in aggregate variables and returns inputs to heterogeneous agents block
function G_hetinput(Xlag,X,XPrime,RShock_,np::NumericalParameters)

	@unpack mp = np
	@unpack μ, κ, B, θ_π, Z_ss, r_target = mp

	#unpack variables
	πip, Yp, wp = XPrime
	πi, Y, w = X
	πi_, Y_, w_ = Xlag 

	#get nominal and real rates
	i_ = (r_target +  θ_π*πi_ + RShock_)
	r  = (1+i_)/(1+πi) - 1
	
	#net transfer level T_level
	T_level = Y*(1-w/Z_ss) - r*B

	return [w,r,T_level]

end

# ╔═╡ a739d105-e648-420b-97b5-435532875218
md"""
In addition to that, we just need a now smaller set of aggregate equations characterizing the equilibrium:
"""

# ╔═╡ 2e4d93d6-c328-4281-8844-c57a085d50e3
function G2_fun(Xlag,X,XPrime,het_out,RShock,np::NumericalParameters)

	@unpack mp = np
	@unpack μ, κ, B, θ_π, Z_ss, r_target = mp

	#unpack stuff:
	πip, Yp, wp = XPrime
	πi, Y, w = X
	πi_, Y_, w_ = Xlag 
	
	L, A = het_out

	#helper variable to get correct eltype for container below
	ind_var = Yp + Y + Y_ + L + RShock

	#get nominal and real rates
	i = (r_target +  θ_π*πi + RShock)
	rp = (1+i)/(1+πip) - 1

	#container for residuals
	eq_out = zeros(eltype(ind_var),length(Xlag)) 

	#NK Philips curve
	eq_out[1] +=  κ*(w/Z_ss - (1/μ)) + 1/(1+rp)*(Yp/Y)*log(1+πip) - log(1+πi)

	#labor market clearing
	eq_out[2] += Y - Z_ss*L

	#Asset market clearing
	eq_out[3] += A - B

	return eq_out

end

# ╔═╡ 84dec990-0264-4d5d-994f-87cdd7e97a42
md"""
With that, we can proceed as in the previous section. Instead of just the selection matrix `P` there are a few more matrices featuring derivatives of `G_hetinput` that need to be multiplied with the HA Block Jacobian, which now also appears in the expression for $F_Z$. Overall, though, everything works according to the same principle. 
"""

# ╔═╡ d7530a99-2c7c-4cde-8263-5707b8cd1974
function get_jacobs_G2(Js,SS_objs)

	np = SS_objs.np

	#define vectors of steady state variables
	Xss = [0.0,SS_objs.Y_ss,SS_objs.w_ss]
	het_out_ss = [SS_objs.L_ss,SS_objs.A_ss]

	#check that equations hold in SS
	check = G2_fun(Xss,Xss,Xss,het_out_ss,0.0,np)
	if maximum(abs.(check)) > 1e-8
		@warn "large F residuals"
	end

	#compute derivatives of G2
	dG2_X = ForwardDiff.jacobian(x ->
					G2_fun(Xss,x,Xss,het_out_ss,0.0,np),Xss)
	
	dG2_Xlag = ForwardDiff.jacobian(x -> 
					G2_fun(x,Xss,Xss,het_out_ss,0.0,np),Xss)
	
	dG2_XPrime = ForwardDiff.jacobian(x -> 
					G2_fun(Xss,Xss,x,het_out_ss,0.0,np),Xss)
	
	dG2_het = ForwardDiff.jacobian(x -> 
					G2_fun(Xss,Xss,Xss,x,0.0,np),het_out_ss)
	
	dG2_RShock = ForwardDiff.derivative(x -> 
					G2_fun(Xss,Xss,Xss,het_out_ss,x,np),0.0)
	
	#Note: derivative w.r.t. XPrime not needed

	#derivatives of HA block inputs 
	dG_hi = ForwardDiff.jacobian(x -> G_hetinput(Xss,x,Xss,0.0,np),Xss)
	
 	dG_hilag = ForwardDiff.jacobian(x -> G_hetinput(x,Xss,Xss,0.0,np),Xss)
	
	dG_hi_Rshocks = ForwardDiff.derivative(x ->
  								G_hetinput(Xss,Xss,Xss,x,np),0.0)

	#define some helper matrices
	Im = sparse(I,np.T,np.T)
	Im_ = spdiagm(-1=>ones(np.T-1))
	Imp = spdiagm(1=>ones(np.T-1))

	#get J into suitable shape (again, note ordering of variables)
	J_reshaped = [hcat(Js[:,2]...) ; hcat(Js[:,1]...)]
	
	#assemble Aggregate Jacobians
	F_x = kron(dG2_het,Im)*J_reshaped*(kron(dG_hi,Im) .+ kron(dG_hilag,Im_)) .+ 
				kron(dG2_X,Im) .+ kron(dG2_XPrime,Imp) .+ kron(dG2_Xlag,Im_)

	F_RShock = kron(dG2_het,Im)*J_reshaped*kron(dG_hi_Rshocks,Im_) .+
					kron(dG2_RShock,Im)

	
	return F_x, F_RShock

end

# ╔═╡ d1f4ce92-1bd4-42ef-b4ea-e37cdb117579
md"""
As you may expect, I will again compare the resulting $F_X$, $F_Z$ and inflation response to the DAG method.
"""

# ╔═╡ 57b4a641-8cf0-49ad-bc4c-c85e7071e355
begin 

F_x4, F_RShock4= get_jacobs_G2(Js,SS_objs)

println("Maximum difference for F_x: ",maximum(abs.(F_x1 .- F_x4)))
println("Maximum difference for F_RShock: ",maximum(abs.(F_RShock1 .- F_RShock4)))

#get aggregate variables
GEJ = -F_x4\Matrix(F_RShock4)
AggVars_dyn2 = GEJ*dRShock

#extract inlfation response
π_dyn2 = AggVars_dyn2[1:300,:]

println("Max. abs. diff. π compared to DAG: ", maximum(abs.(π_dyn2 .- π_dag)))

#plot inflation response
plot(1:20,10_000*π_dyn2[1:20,:],  labels = labels, linewidth = 3.0, ylabel = "Basis Points", legend = :topright,title = "Inflation Response (G method 2)")

end

# ╔═╡ 76149d26-923a-4cd0-b3f6-285d1e00cb85
md"""
Everything is the same, as one would hope. And getting the aggregate Jacobians is now essentially as fast as for the DAG method:
"""

# ╔═╡ e7a61ca2-b9a3-4dae-97d6-e7d1046acaa2
@btime get_jacobs_G2($Js,$SS_objs);

# ╔═╡ e5a48ca3-c3b2-423a-8823-dd672941fd58
md"""
Finally, you may ask yourself that even though we now got the GE Jacobians for $Y$, $w$ and $\pi$, what about the other variables? Do we still need to do DAG-ing for that?

Of course not, we can again just write a function that gives us the remaining variables as functions of $Y$, $w$ and $\pi$ and the shocks, differentiate it and combine it with the GE Jacobian we already have. This is done by the two functions below:
"""

# ╔═╡ 03eb6ef0-6b44-4d2d-a972-fff171c22772
#takes in small set of aggr variables and shocks and returns all aggregate variables
function AggVars(Xlag,X,RShocks,np)

	@unpack mp = np
	@unpack μ, κ, B, θ_π, Z_ss, r_target = mp

	#unpack stuff
	πi, Y, w = X
	πi_, Y_, w_ = Xlag 

	RShock_, RShock = RShocks

	#get aggregates 

	#nominal rates
	i = (r_target +  θ_π*πi + RShock)
	i_ = (r_target +  θ_π*πi_ + RShock_)

	#real rate
	r  = (1+i_)/(1+πi) - 1

	#taxes
	τ = r*B

	#labor
	L = Y/Z_ss

	#dividend
	div = Y - w*L

	#net transfer
	T_level = div - τ

	return [w,r,T_level,div,i,Y,τ,πi]

end

# ╔═╡ d9fd8c94-f1a2-4628-b93f-51361f6a3efa
function assemble_G_Aggr(GEJ,SS_objs,np)

	#define vectors of steady state variables
	Xss = [0.0,SS_objs.Y_ss,SS_objs.w_ss]
	het_out_ss = [SS_objs.L_ss,SS_objs.A_ss]

	#derivatives of aggregate variables w.r.t variables
	dAggr_X = ForwardDiff.jacobian(x -> AggVars(Xss,x,zeros(2),np),Xss)
	dAggr_Xlag = ForwardDiff.jacobian(x -> AggVars(x,Xss,zeros(2),np),Xss)
	dAgg_RShocks = ForwardDiff.jacobian(x -> AggVars(Xss,Xss,x,np),zeros(2))

	#define the helper matrices
	Im = sparse(I,np.T,np.T)
	Im_ = spdiagm(-1=>ones(np.T-1))
	Imp = spdiagm(1=>ones(np.T-1))

	d_Aggr_X = kron(dAggr_X,Im) .+ kron(dAggr_Xlag,Im_)

	G_Aggr = (kron(dAgg_RShocks[:,2],Im) .+ kron(dAgg_RShocks[:,1],Im_) .+ d_Aggr_X*GEJ)

	 return G_Aggr

end

# ╔═╡ 3107d9de-b4ec-4954-8f09-4e1896c83f62
md"""
For example, we can now also plot the response of the nominal rate:
"""

# ╔═╡ 20688c81-1b2c-4a1e-bfb6-38f65c0e2d6f
begin
G_Aggr = assemble_G_Aggr(GEJ,SS_objs,np)
AggVars_dyn3 = G_Aggr*dRShock

#extract nominal rate response
i_dyn3 = AggVars_dyn3[1201:1500,:]

#plot nom. rate response
plot(1:20,10_000*i_dyn3[1:20,:],  labels = labels, linewidth = 3.0, ylabel = "Basis Points", legend = :topright, title = "Nom. rate Response (G method 2)")

end

# ╔═╡ d1788944-9c32-4e98-b60e-b8834dba9dde
md"""
We see that if the accommodative nominal interest rate shock is really persistent, inflation may react so much that the Central Bank is induced to actually *raise* the nominal rate upon impact of the shock. The *real* rate still decreases on impact  though:
"""

# ╔═╡ 0933d143-a15b-49d9-8ced-9ddc42526a48
begin
#extract nominal rate response
r_dyn3 = AggVars_dyn3[301:600,:]

#plot nom. rate response
plot(1:20,10_000*r_dyn3[1:20,:],  labels = labels, linewidth = 3.0, ylabel = "Basis Points", legend = :topright, title = "Real rate Response (G method 2)")
end

# ╔═╡ 2d6e16f7-a39f-400a-8645-91048ed7e120
md"""
(Yes, these responses look somewhat ugly. Feel free to convince yourself that the ones in the ABRS notebook look the same.)
"""

# ╔═╡ d03ad22d-5543-4105-aa8d-dfe2ee180050
md"""
Overall, I feel this method provides a good compromise between the speed of the DAG method and the convenience of the Dynare-like method. 

A downside compared to the former is that it is perhaps not as straightforward to use what ABRS call "Solved Blocks", although it can be adapted to some extent: If it were desirable to have one such "block" in a richer version of this model, one could e.g. add an additional set of inputs named `solved_out` to `G2_fun` and then proceed similar to the HA block.
"""

# ╔═╡ 84f43682-fb05-4b0e-aa37-c12819b4f666

md"""
## Conclusion

I hope you found this notebook useful. If yes, please consider starring its [Github repo](https://github.com/mhaense1/SSJ_Julia_Notebook) and share it with colleagues who might also find it useful. Feedback and suggestions are also very welcome.

Finally, since you seem interested in Heterogeneous Agent Macro, have a look at my [Personal Website](https://mhaense1.github.io/) to learn about my research in this area and/or to get in contact with me.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BasicInterpolators = "26cce99e-4866-4b6d-ab74-862489e035e0"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NonlinearSolve = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuantEcon = "fcd29c91-0bd7-5a09-975d-7ac3f643a60c"
Setfield = "efcf1570-3423-57d1-acb7-fd33fddbac46"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
BasicInterpolators = "~0.7.1"
BenchmarkTools = "~1.6.0"
ForwardDiff = "~1.0.1"
NonlinearSolve = "~4.8.0"
Parameters = "~0.12.3"
Plots = "~1.40.13"
PlutoUI = "~0.7.62"
QuantEcon = "~0.16.6"
Setfield = "~1.1.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "12c363485656ee882962c8c6a2626f76ca441e68"

[[deps.ADTypes]]
git-tree-sha1 = "e2478490447631aedba0823d4d7a80b2cc8cdb32"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.14.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

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
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "9606d7832795cbef89e06a550475be300364a8aa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.19.0"

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

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

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

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

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
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

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

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0df00546373af8eee1598fb4b2ba480b1ebe895c"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.10"

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
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "ConcreteStructs", "DataStructures", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "FastPower", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "Parameters", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "SciMLStructures", "Setfield", "Static", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "TruncatedStacktraces"]
git-tree-sha1 = "1bcd3a5c585c477e5d0595937ea7b5adcda6c621"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.174.0"

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
git-tree-sha1 = "c8d85ecfcbaef899308706bebdd8b00107f3fb43"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.6.54"

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

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3e6d038b77f22791b8e3472b7c633acea1ecac06"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.120"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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
git-tree-sha1 = "7d7822a643c33bbff4eab9c87ca8459d7c688db0"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.11"
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

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "7de7c78d681078f027389e067864a8d53bd7c3c9"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

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
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

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
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

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
git-tree-sha1 = "4424dca1462cc3f19a0e6f07b809ad948ac1d62b"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.16"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d7ecfaca1ad1886de4f9053b5b8aef34f36ede7f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.16+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "fee60557e4f19d0fe5cd169211fdda80e494f4e8"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.84.0+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "3169fd3440a02f35e549728b0890904cfd4ae58a"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.12.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f93655dc73d7a0b4a368e3c0bce296ae035ad76e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.16"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

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

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

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

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

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
git-tree-sha1 = "4f34eaabe49ecb3fb0d58d6015e32fd31a733199"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.8"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

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
weakdeps = ["LineSearches"]

    [deps.LineSearch.extensions]
    LineSearchLineSearchesExt = "LineSearches"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

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

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
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

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "b14c7be6046e7d48e9063a0053f95ee0fc954176"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.9.1"

[[deps.NLopt]]
deps = ["CEnum", "NLopt_jll"]
git-tree-sha1 = "35a8d661041aa6a237d10e12c29a7251a58bf488"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "1.1.4"

    [deps.NLopt.extensions]
    NLoptMathOptInterfaceExt = ["MathOptInterface"]

    [deps.NLopt.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b0154a615d5b2b6cf7a2501123b793577d0b9950"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.10.0+0"

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
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

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

[[deps.Optim]]
deps = ["Compat", "EnumX", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "31b3b1b8e83ef9f1d50d74f1dd5f19a37a304a1f"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.12.0"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "275a9a6d85dc86c24d03d1837a0010226a96f540"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.3+0"

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
git-tree-sha1 = "809ba625a00c605f8d00cd2a9ae19ce34fc24d68"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.13"

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
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "d3de2694b52a01ce61a036f18ea9c0f61c4a9230"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.62"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "6f7cd22a802094d239824c57d94c8e2d0f7cfc7d"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.18"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "555c272d20fc80a2658587fb9bbda60067b93b7c"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.19"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

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

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "eb38d376097f47316fe089fc62cb7c6d85383a52"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "2766344a35a1a5ec1147305c4b343055d7c22c90"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.QuantEcon]]
deps = ["DSP", "DataStructures", "Distributions", "FFTW", "Graphs", "LinearAlgebra", "Markdown", "NLopt", "Optim", "Pkg", "Primes", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test"]
git-tree-sha1 = "034293b29fdbcae73aeb7ca0b2755e693f04701b"
uuid = "fcd29c91-0bd7-5a09-975d-7ac3f643a60c"
version = "0.16.6"

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

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "86a8a8b783481e1ea6b9c91dd949cb32191f8ab4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.15"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "2fd047893cb0089b180fcbb7e8434ba15dcc2841"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.87.0"

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

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

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

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

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

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

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
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "b81c5035922cc89c2d9523afc6c54be512411466"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.5"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "8e45cecc66f3b42633b8ce14d431e8e57a3e242e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

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

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

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
git-tree-sha1 = "2d529b6b22791f3e22e7ec5c60b9016e78f5f6bf"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.4"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "3748bd928e68c7c346b52125cf41fff0de6937d0"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.29"

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
git-tree-sha1 = "d62610ec45e4efeabf7032d67de2ffdea8344bed"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.1"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

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
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "XML2_jll"]
git-tree-sha1 = "49be0be57db8f863a902d59c0083d73281ecae8e"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.23.1+0"

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
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

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
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

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
git-tree-sha1 = "9caba99d38404b285db8801d5c45ef4f4f425a6d"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.1+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

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
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

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
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

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
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3cad2cf2c8d80f1d17320652b3ea7778b30f473f"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.3.0+0"

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
git-tree-sha1 = "002748401f7b520273e2b506f61cab95d4701ccf"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.48+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "c950ae0a3577aec97bfccf3381f66666bc416729"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.8.1+0"
"""

# ╔═╡ Cell order:
# ╟─19ea0f40-4b3e-11ef-1820-4992f2ba942f
# ╠═9bae8cdb-18bb-4c5b-8984-c27fd9417874
# ╠═a7d93065-0d0c-45ac-b497-79845fccf27a
# ╟─85614fee-9eea-4b9a-ac00-91847d8d7cce
# ╠═e42b4882-d822-43f7-8bec-4ebaffb5d17b
# ╟─7097a48b-8906-4197-8012-1b3673ab1d78
# ╟─d0385d4e-08ca-43b1-ae70-8da38175cb6e
# ╠═dbf4d282-2c04-4d9d-b1f1-2ad8f6e9098b
# ╟─5f99a5dd-abab-4fd4-bb67-0ba7dd201977
# ╠═7e4416f4-1211-45bc-9511-e17028af154b
# ╠═3708fb95-a65d-4a1f-bb5e-f4db9a8766b9
# ╟─78f02c49-becd-4060-9f2f-ac250f468110
# ╠═60fa7af4-1bd9-4acb-bfe0-61801317b177
# ╟─3e960018-54bf-4331-9ac0-e130e2393778
# ╠═dfb79d2c-d29c-4e27-b677-959b3d7b57f1
# ╟─b18c850d-1edd-4d30-b67f-5c6a21f2a7fc
# ╠═3dc4a1f0-7d85-42c9-a8e3-4de80f604b1b
# ╟─cd3b00b6-cfc5-4096-8c63-601ae7cd214e
# ╟─b1d47ec8-4958-449a-b9df-9f4cca062ed9
# ╟─12e89139-f672-4265-bd5a-5639e3f96a5f
# ╟─7ea684fb-e2a5-4935-b18e-f491b4c098b4
# ╠═42e36f72-e02c-435f-8fa6-30309b3ee3e6
# ╠═a40334ec-d57f-477e-97ad-9aa82b7ff46d
# ╠═1e2773ef-0dd5-4c6e-ae9c-93001fd2e0ff
# ╟─54402a9f-56df-439f-98b8-64d15ad0705e
# ╠═2f8230fd-a3b6-42c9-8ebc-fed90f15733e
# ╟─d9e401e9-b253-425f-9d98-340088757c2a
# ╠═acf9f7a7-300f-423d-8220-6014327ec39b
# ╟─9c53d4a7-526d-4065-88b5-0ef5e04f9b62
# ╠═fed0b8e1-fe6c-4c17-8c03-bc819ac7c45d
# ╠═575be7fa-ae2f-4fac-b7b7-c6ae6a1fec93
# ╟─ca2de9d9-879f-46ac-a02e-b5af6ad20f6f
# ╠═0303d80f-a072-4a51-bfaf-d37f721fe82f
# ╟─e9772d25-e08b-4888-833c-1a3e79518b22
# ╠═dac31547-cebf-4c1b-be47-5253b4b86abc
# ╟─6da15895-81a6-434e-9002-522a3b3348a1
# ╟─7a2bba2b-7084-4979-82c8-6cf3e5daec3d
# ╠═fb08ff71-d0f9-4065-99ec-e4a1f6479072
# ╟─3e4d39b1-34b2-4a63-a081-49db00e00866
# ╠═f7537f35-7549-4cdc-994d-616394084d3c
# ╟─c23a83c1-4009-4378-b229-27b56cf4f640
# ╠═90c792f4-eef4-4900-813d-b6dc29e8389c
# ╟─06413e4c-9b9e-4360-9222-3449bb751418
# ╠═12e86c4c-dbae-4095-9c79-293ba38df5dc
# ╠═a79aebd9-44b4-4267-87bc-62b8c8723766
# ╟─e1ec7322-f8fc-428f-85db-04ccbabf023f
# ╠═24bdcddd-af1f-430c-bed5-3cda6536f4a1
# ╠═ffc02cf0-d10f-4dcb-a33a-9705ac6ccb12
# ╟─3ca6f293-b683-4c70-b9eb-93fff236b786
# ╠═404e75c6-77d3-48df-9376-4eaae9cc9827
# ╠═1a52ce21-ada0-4dea-b484-e6c4393328d4
# ╟─6555e425-097c-45d3-9fc2-6c579ca099d5
# ╟─62e7a938-89b2-4123-b431-1f4cd98b3da8
# ╠═17da4c85-a1c8-4204-91df-001a17bd07c4
# ╟─0610663b-a8a9-4182-86eb-98093e77d35b
# ╠═2e128a66-9a3b-460a-9641-2bb8cba9c6b3
# ╟─09f07dd5-a950-44f7-8a20-6070a8fa45b3
# ╠═48e1498e-2d8b-4051-b90c-799ab37c3338
# ╟─6f160a63-8fec-4f8b-a2ea-0dc691e30d30
# ╠═fb7958ac-014b-40cb-9020-7d6f645a7f87
# ╟─b2897cf6-720a-45b5-860d-85044cdd7f28
# ╠═53188374-85e0-41f5-aa3b-a89a339cd974
# ╠═7de8f598-dd30-428a-afb6-4cc372e1c4f7
# ╟─9790e1b7-cb54-4086-ba92-4006667359b0
# ╟─331dfd8e-c4ad-4c6d-89c9-691eb5a16f0c
# ╠═f0cdbfc2-6216-491b-a874-0313e20558b4
# ╟─a739d105-e648-420b-97b5-435532875218
# ╠═2e4d93d6-c328-4281-8844-c57a085d50e3
# ╟─84dec990-0264-4d5d-994f-87cdd7e97a42
# ╠═d7530a99-2c7c-4cde-8263-5707b8cd1974
# ╟─d1f4ce92-1bd4-42ef-b4ea-e37cdb117579
# ╠═57b4a641-8cf0-49ad-bc4c-c85e7071e355
# ╟─76149d26-923a-4cd0-b3f6-285d1e00cb85
# ╠═e7a61ca2-b9a3-4dae-97d6-e7d1046acaa2
# ╟─e5a48ca3-c3b2-423a-8823-dd672941fd58
# ╠═03eb6ef0-6b44-4d2d-a972-fff171c22772
# ╠═d9fd8c94-f1a2-4628-b93f-51361f6a3efa
# ╟─3107d9de-b4ec-4954-8f09-4e1896c83f62
# ╠═20688c81-1b2c-4a1e-bfb6-38f65c0e2d6f
# ╟─d1788944-9c32-4e98-b60e-b8834dba9dde
# ╠═0933d143-a15b-49d9-8ced-9ddc42526a48
# ╟─2d6e16f7-a39f-400a-8645-91048ed7e120
# ╟─d03ad22d-5543-4105-aa8d-dfe2ee180050
# ╟─84f43682-fb05-4b0e-aa37-c12819b4f666
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
