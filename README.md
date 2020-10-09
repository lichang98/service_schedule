# Serevice Schedule

This project is written for competition https://tianchi.aliyun.com/competition/entrance/531831/information

## Project Structure

The project contains some methods and utils for scoring, data loading and saving.
The files listed below are some methods that have tried.

* spt_benchmark.cpp: This is a simply policy-based method, which try assign current waitting task to best suit expert. Also, in order to make the expert load balance, the expert will be sorted by their processing speed, so far workload and number of available channels. If no available suit experts(processing time != 999999), the task will wait.
* monte_carlo_ts.cpp: using monte carlo tree search method(UCT). The detail procedure is available at the top of the method `run_alg` of this file. Note that the file using same method with the below one, but this file has memory leak, and the below file solved the problem.
* mcts.cpp: Same method as above one.
* greedy2.cpp: In this file, we conclude the operations into three type: `assign`, `migrate` and `swap`. A task can assign to suitable expert if available, if not, it can choose assign to not suitable expert or wait(which can be randomly decided). At each iteration, the tasks on not suitable experts will check if current time suitable experts available, if so, the task can migrate to suitable expert to execute. At some special cases, there may exist a extream case when **expert e_a has tasks want to migrate to expert e_b and expert e_b has tasks want to migrate to e_c and expert e_c want to migrate to e_a**, which forms a depedency cycle and stuck into a deadlock. Since adding cycle detection in each iteration is time consuming, so I add a `swap` operation, if for two task all on their not suitable experts, and at least  one part expert is suitable for another, they can swap. The above three operations can all be randomly taken. A task can choose assin or not, choose migration or not and choose swap or not at each iteration.
* greedy.cpp: Since the dimension is too huge for above method, so i want to add `snapshot` for a fine solution. For example, a solution may take 3000 time slots to finish, if the solution is good, i can take `snapshot` at time slot 2000, 2400, 2800 etc. and then start random search process from the snapshot, then the random search space will decrease a lot.
* ga.cpp: Since each task has max migration count *M*, we can pre decide the experts the task will bypass, and must keep sure the last one is suitable and no repetation for two adjacent. For example [-1,-1,3,89,3] means the tasks by pass expert idx 3 89 and 3, where 3 is allowed to present more than once, but not consecutive. Well, I forget taking waitting time after task generation time and pirority for tasks, these can be add into it. During the running of the algorithm, time marker will preset for each tasks, the tasks will decrease the available spaces of time marker list for each expert at corresponding time.

The files listed below are for scoring, data loading and saving and entities definitions.

* utils.hpp: this file and the file below are initial version of definitions of entities, scoring and data loading/saving.
* metrics.hpp: initial version of scoring
* monte_utils.hpp: Later version of definition of entities, scoring and data loading/saving
* monte_metrics.hpp: Later version of scoring
