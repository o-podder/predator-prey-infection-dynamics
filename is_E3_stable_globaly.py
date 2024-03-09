import predator_prey_system as S
from random import choice


def check_E3_stable_iteration_correctness(): 
    ps = S.iterate_params_E3_stable(b2_step=0.2, b4=1)
    for i, p in enumerate(ps): 
        p = S.Parameters(p)
        assert  p.steady_state_E3_exists()
        assert  not p.is_E3_steady_state_inner_eigen_positive() 
        assert  p.is_steady_state_E3_stable_as_2d()

#check_E3_stable_iteration_correctness()


def hypothesis_E3_stable_globaly(): 
    # When E3 exists and stable as 3d, does it attract 
    # all inner trajectories? 

    ps = S.iterate_params_E3_stable(b2_step=0.1, alpha_step=0.2, beta_step=0.2)
    failed = []
    for i, p in enumerate(ps): 
        print(i)
        s = S.PredatorPreySystem(p)
        E3 = s.params.steady_state_E3()
        p_failed = s.is_global_limit_state(E3, depth=1)
        if p_failed is not None: 
            print('fail', p)
            failed.extend(
                ((p, x0) for x0 in p_failed)
            )

    for _ in range(min(len(failed), 20)): 
        p, x0 = choice(failed)
        s = S.PredatorPreySystem(p)
        S.plot(s, x0, 3000, title=str([p, x0]))

    return failed


failed = hypothesis_E3_stable_globaly()


#p = [0.2, 8.9182, 0.1, 1, 0.8]
#s = S.PredatorPreySystem(p)
#E3 = s.params.steady_state_E3()
#x0 = [0.3, 0.2, 0.403]
#b = s.is_limit_state(E3, x0, depth=3)
#print(b)
#S.plot(s, x0, 1000, state=E3, title=str(x0))


#p = [0.2, 8.9182, 0.1, 1, 0.8]
#s = S.PredatorPreySystem(p)
#E3 = s.params.steady_state_E3()
#failed = s.is_global_limit_state(E3, depth=1)
# if failed: 
    #for x0 in failed: 
    #    S.plot(s, x0, 1000, state=E3, title=str(x0))
