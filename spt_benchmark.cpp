/**
 * This file is using simple SPT strategy as the benchmark
 * The shortest processing time rule orders the jobs in the order of increasing processing times.
 * Whenever a machine is freed, the shortest job ready at the time will begin processing.
 */

#include "metrics.hpp"
#include <vector>
#include <algorithm>
#include <ctime>

// Group the tasks by their type
// Each row corresponds to one type
std::vector<std::vector<utils::Task *>> task_groupby_type(int num_types, std::vector<utils::Task> &tasks)
{
    std::vector<std::vector<utils::Task *>> group_tasks(num_types);
    for (int i = 0; i < tasks.size(); ++i)
    {
        int task_type = tasks[i].type;
        group_tasks[task_type].push_back(&tasks[i]);
    }
    return group_tasks;
}

// Sort the order of tasks in each group by the generating time and response time
void task_sort_each_group_by_tm_resptm(std::vector<std::vector<utils::Task *>> &group_tasks)
{
    int size = group_tasks.size();
    for (int i = 0; i < size; ++i)
    {
        std::sort(group_tasks[i].begin(), group_tasks[i].end(), [](utils::Task *a, utils::Task *b) -> bool {
            if (a->tm_stamp != b->tm_stamp)
                return a->tm_stamp < b->tm_stamp;
            else if (a->max_resp_tm != b->max_resp_tm)
                return a->max_resp_tm < b->max_resp_tm;
            else
                return a->task_id < b->task_id;
        });
    }
}

// Group experts by their types
// The return result is a two dimension vector, each row corresponds to one type, and each value in a row is the expert index
std::vector<std::vector<int>> expert_group_by_type(int num_types, std::vector<utils::Expert> &experts)
{
    std::vector<std::vector<int>> group_experts(num_types);
    for (int i = 0; i < experts.size(); ++i)
    {
        for (int j = 0; j < experts[i].process_dura.size(); ++j)
        {
            if (experts[i].process_dura[j] < utils::EXPERT_NOT_GOOD_TIME)
            {
                group_experts[j].push_back(i);
            }
        }
    }
    return group_experts;
}

// Sort experts in each group by their processing time
void expert_sort_each_group_by_processtm(std::vector<std::vector<int>> &group_experts, std::vector<utils::Expert> &experts)
{
    for (int i = 0; i < group_experts.size(); ++i)
    {
        std::sort(group_experts[i].begin(), group_experts[i].end(), [experts, i](int &a, int &b) -> bool {
            return experts[a].process_dura[i] < experts[b].process_dura[i];
        });
    }
}

// Shortest Process Time strategy
// Tasks are grouped by type and sorted by their allowed max reponse time
// Experts are also grouped by types which they good at, each expert may belong to multiple groups
// In algorithm, tasks will be fetched from group tasks and assign to corresponding type group experts
// If experts are all busy, the tasks need to wait.
std::vector<std::vector<int>> spt_run(std::vector<utils::Task> &tasks, std::vector<utils::Expert> &experts,
                                      std::vector<std::vector<utils::Task *>> &group_tasks,
                                      std::vector<std::vector<int>> &group_experts)
{
    std::vector<std::vector<int>> result; // each line contains three value: task id, expert id, task start processing time
    int num_left_tasks = tasks.size();
    std::vector<int> task_group_progresses(group_tasks.size());
    for (int i = 0; i < task_group_progresses.size(); ++i)
        task_group_progresses[i] = 0;
    int env_tm = 0; // time slots
    printf("Start assigning tasks to experts...\n");
    printf("Total number of tasks=%d\n", num_left_tasks);
    while (num_left_tasks > 0)
    {
        for (int i = 0; i < task_group_progresses.size(); ++i)
        {
            if (task_group_progresses[i] < group_tasks[i].size())
            {
                // Try assign task to expert
                utils::Task *curr_task = group_tasks[i][task_group_progresses[i]];
                int task_type = curr_task->type;
                for (int expt_idx : group_experts[task_type])
                {
                    if (experts[expt_idx].assign_task(curr_task))
                    {
                        // Successful assign this task to the expert
                        task_group_progresses[i]++;
                        curr_task->start_process_tmpt = env_tm;
                        result.emplace_back(std::vector<int>({curr_task->task_id, experts[expt_idx].id, env_tm}));
                        break;
                    }
                }
            }
        }
        env_tm++;
        // Put forward one time slot
        for (utils::Expert &expt : experts)
        {
            std::vector<utils::Task *> finish_tasks = expt.update(env_tm);
            num_left_tasks -= finish_tasks.size();
        }
    }
    return result;
}

int main(int argc, char const *argv[])
{
    std::vector<utils::Task> tasks = utils::load_work_order();
    std::vector<utils::Expert> experts = utils::load_expert_process_duras();
    int num_types = experts[0].num_types();
    std::vector<std::vector<utils::Task *>> group_tasks = task_groupby_type(num_types, tasks);
    task_sort_each_group_by_tm_resptm(group_tasks);
    std::vector<std::vector<int>> group_experts = expert_group_by_type(num_types, experts);
    expert_sort_each_group_by_processtm(group_experts, experts);
    // Start running, assign tasks and process
    std::vector<std::vector<int>> result = spt_run(tasks, experts, group_tasks, group_experts);
    // save result
    time_t date = time(nullptr);
    tm *date_tm = localtime(&date);
    char result_tm_stamp[50];
    sprintf(result_tm_stamp, "%02d%02d%02d_%02d%02d%02d.csv", date_tm->tm_year + 1900 - 2000, date_tm->tm_mon + 1, date_tm->tm_mday, date_tm->tm_hour, date_tm->tm_min, date_tm->tm_sec);
    utils::save_result(strcat(utils::PRED_RESULT_PREFIX, result_tm_stamp), result);
    // Calculating Scores
    std::vector<double> expt_workloads(experts.size(), 0), task_resp_tmout(tasks.size(), 0), exec_eff(tasks.size(), 0);
    for (int i = 0; i < tasks.size(); ++i)
    {
        task_resp_tmout[i] = metrics::task_response_timeout(tasks[i]);
        exec_eff[i] = metrics::task_exec_efficiency(tasks[i]);
    }
    for (int i = 0; i < experts.size(); ++i)
        expt_workloads[i] = metrics::expert_workload(experts[i]);
    double score = metrics::score(expt_workloads, task_resp_tmout, exec_eff);
    printf("Score=%lf\n", score);
    return 0;
}
