# ROADMAP
1. Add a actor configuration `auto_num_gpus`.
2. In ArRay, we will set each actor's `num_gpus` automatically and design a policy to reduce GPU fragment.
3. We collect the memory usage and GPU utilization by running this actor alone, `num_gpus = min(mem_usage, gpu_util)`.
4. Find out an optimal actor placement combination to reduce GPU fragment.

