# kinfast on the MuJoCo Menagerie

Real production MJCF models from google-deepmind/mujoco_menagerie.
parse = kinfast loads the model XML. oracle = full model directory
downloaded and FK compared body-by-body against MuJoCo's mj_forward.
Reproduce: `python examples/menagerie.py --fetch` then run again.

| model | tier | dof | result |
|---|---|---|---|
| franka_emika_panda | parse | 9 | loads |
| franka_fr3 | parse | 7 | loads |
| universal_robots_ur5e | oracle | 6 | max FK diff vs MuJoCo 3.0e-07 |
| universal_robots_ur10e | parse | 6 | loads |
| kuka_iiwa_14 | parse | 7 | loads |
| kinova_gen3 | parse | 7 | loads |
| ufactory_xarm7 | parse | 13 | loads |
| rethink_robotics_sawyer | parse | 7 | loads |
| unitree_go2 | parse | 12 | loads |
| unitree_g1 | parse | 29 | loads |
| anybotics_anymal_c | parse | 12 | loads |
| boston_dynamics_spot | parse | 12 | loads |
| agility_cassie | parse | | FAILED: ValueError: body left-achilles-rod: ball joints not supported |
| shadow_hand | parse | 24 | loads |
| robotiq_2f85 | parse | 8 | loads |
| trs_so_arm100 | oracle | 6 | max FK diff vs MuJoCo 3.2e-07 |
| aloha | parse | 16 | loads |
| hello_robot_stretch | parse | 17 | loads |
| google_barkour_vb | parse | 12 | loads |
| pal_tiago | parse | 22 | loads |
