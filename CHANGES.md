Changes in some implementations from Michalis's code

- The analytical bunch length calculation from Michalis can/will be replaced by the statistical one on `xpart.Particles` objects. Analytical shows weaknesses when bucket gets full. Only makes sense if we are doing tracking (need the particles)! Could keep the analytical calculation for emittances calculations based on Nagaitsev integrals (for users who don't track).
- 