from metaworld.benchmarks import ML1
from PIL import Image
from mujoco_py.generated import const

tasks = ML1.available_tasks()
for task in tasks:
    env = ML1.get_train_tasks(task)  # Create an environment with task `pick_place`
    tasks = env.sample_tasks(1)  # Sample a task (in this case, a goal variation)
    env.set_task(tasks[0])  # Set task

    obs = env.reset()  # Reset environment
    im = Image.fromarray(env.render(mode='rgb_array'))
    viewer =  env._task_envs[0].viewer
    #viewer.cam.trackbodyid = 0         # id of the body to track ()
    #viewer.cam.distance = self.model.stat.extent * 1.0         # how much you "zoom in", model.stat.extent is the max limits of the arena
    viewer.cam.trackbodyid = 0
    #viewer.cam.lookat[0] = 0.2
    #viewer.cam.lookat[1] = 0.75
    viewer.cam.lookat[2] = 0.01
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -25
    viewer.cam.azimuth = -110
    viewer.cam.trackbodyid = -1

    #env._task_envs[0].viewer.cam.type = const.CAMERA_FIXED
    #env._task_envs[0].viewer.cam.fixedcamid = 0
    #a = env.action_space.sample()  # Sample an action
    #obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    #env.action_space.sample()
    #while True:
    #    env.render()
    #    print(viewer.cam.elevation)
    #    print(viewer.cam.azimuth)
    #    print('---')
    rgb = env.render(mode = 'rgb_array', width=1500, height=1500)
    im = Image.fromarray(rgb)
    im.save("%s.jpg"%task)
    env.close()
