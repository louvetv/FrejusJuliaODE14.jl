module FrejusJuliaODE14

export rk4_step, euler_step, solve_ode

"""
    solve_ode
    generic solution of the ODE du/dt = f(u, p, t) using a chosen time-stepping function.
"""
function solve_ode(f, u0, p, tspan, nsteps::Integer, step::Function = euler_step)
    t0, tf = tspan
    dt = (tf - t0) / nsteps
    t = range(t0, stop=tf, length=nsteps + 1)
    u = Vector{typeof(u0)}(undef, nsteps + 1)
    u[1] = u0

    for n in 1:nsteps
        u[n + 1] = step(f, u[n], p, t[n], dt)
    end

    return t, u
end

function solve_ode(f, u0, p, tspan, nsteps::Integer; step::Function = euler_step)
    return solve_ode(f, u0, p, tspan, nsteps, step)
end

"""
    rk4_step(f, u, p, t,dt)
"""
function rk4_step(f, u, p, t,dt)
    k1 = f(u, p, t)
    k2 = f(u + dt/2 * k1, p, t + dt/2)
    k3 = f(u + dt/2 * k2, p, t + dt/2)
    k4 = f(u + dt * k3, p, t + dt)
    return u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
end

"""
    euler_step(f, u, p, t, dt)

Compute the explicit Euler steps for the ODE du/dt = f(u, p, t) over the time span tspan with nsteps steps.
"""
function euler_step(f, u, p, t, dt)
    return u + dt * f(u, p, t)
end

end # module FrejusJuliaODE14
