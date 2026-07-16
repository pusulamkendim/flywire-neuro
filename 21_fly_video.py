"""
Quick fly walking video — NeuroMechFly v2 with camera rendering.
CPG tripod gait, 2 seconds, side + top view.
"""

import numpy as np
from pathlib import Path
from flygym import Fly, SingleFlySimulation, Camera
from flygym.preprogrammed import get_cpg_biases
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork

RESULTS_DIR = Path(__file__).parent / 'results'

def main():
    timestep = 1e-4
    run_time = 1.5  # seconds
    num_steps = int(run_time / timestep)

    # CPG controller — tripod gait
    cpg = CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=np.ones(6) * 12.0,
        intrinsic_amps=np.ones(6),
        coupling_weights=(get_cpg_biases('tripod') > 0).astype(float) * 10.0,
        phase_biases=get_cpg_biases('tripod'),
        convergence_coefs=np.ones(6) * 20.0,
    )
    preprogrammed_steps = PreprogrammedSteps()
    leg_names = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']

    # Fly with contact visualization
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        init_pose='stretch',
        control='position',
    )

    # Camera — top-right elevated view
    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name='camera_right',
        targeted_fly_names=[fly.name],
        play_speed=0.2,
        window_size=(1280, 720),
        fps=30,
        timestamp_text=True,
        draw_contacts=True,
    )

    sim = SingleFlySimulation(fly=fly, cameras=[cam], timestep=timestep)
    obs, info = sim.reset()

    print(f'Rendering {run_time}s of tripod walking ({num_steps} steps)...')

    for step_i in range(num_steps):
        cpg.step()

        all_joint_angles = []
        all_adhesion = []
        for i, leg in enumerate(leg_names):
            angles = preprogrammed_steps.get_joint_angles(
                leg, cpg.curr_phases[i], cpg.curr_magnitudes[i])
            all_joint_angles.append(angles)
            adhesion = preprogrammed_steps.get_adhesion_onoff(leg, cpg.curr_phases[i])
            all_adhesion.append(adhesion)

        action = {
            'joints': np.concatenate(all_joint_angles),
            'adhesion': np.array(all_adhesion, dtype=np.float64),
        }
        obs, reward, terminated, truncated, info = sim.step(action)
        sim.render()

        if (step_i + 1) % 5000 == 0:
            pct = (step_i + 1) / num_steps * 100
            pos = obs['fly'][0]
            print(f'  {pct:.0f}% — pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})')

    video_path = str(RESULTS_DIR / 'fly_walking.mp4')
    print(f'Saving video: {video_path}')
    cam.save_video(video_path)

    pos = obs['fly'][0]
    print(f'Final position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})')
    print(f'Distance: {np.linalg.norm(pos[:2]):.1f} mm')
    sim.close()
    print('Done!')


if __name__ == '__main__':
    main()
