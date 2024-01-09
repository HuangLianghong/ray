# ROADMAP
1. Add a extra task queue to store tasks that after reconfigured by ArRay.
2. In ArRay, we pre-run each tasks to obtain its GPU resource requirements.
3. The reconfiguration is a binary-search method to test different `num_gpus` of each tasks.
4. After ArRay, we only modify the `num_gpus` of the tasks, and put it back to original task scheduler.

