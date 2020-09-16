// This file is using simple SPT strategy as the benchmark
// The shortest processing time rule orders the jobs in the order of increasing processing times.
// Whenever a machine is freed, the shortest job ready at the time will begin processing.
#include "utils.hpp"
#include <vector>
#include <algorithm>

// Group the tasks by their type
// Each row corresponds to one type
std::vector<std::vector<utils::Task>> task_groupby_type(int num_types, std::vector<utils::Task> tasks)
{
    std::vector<std::vector<utils::Task>> group_tasks(num_types);
    for (int i = 0; i < num_types; ++i)
    {
        int task_type = tasks[i].type;
        group_tasks[task_type].emplace_back(tasks[i]);
    }
    return group_tasks;
}

// Sort the order of tasks in each group by the generating time and response time
void task_sort_each_group_by_tm_resptm(std::vector<std::vector<utils::Task>> &group_tasks)
{
    int size = group_tasks.size();
    for (int i = 0; i < size; ++i)
    {
        std::sort(group_tasks[i].begin(), group_tasks[i].end(), [](utils::Task &a, utils::Task &b) -> bool {
            if (a.tm_stamp != b.tm_stamp)
                return a.tm_stamp < b.tm_stamp;
            else if (a.max_resp_tm != b.max_resp_tm)
                return a.max_resp_tm < b.max_resp_tm;
            else
                return a.task_id < b.task_id;
        });
    }
}

// Group experts by their types
// The return result is a two dimension vector, each row corresponds to one type, and each value in a row is the expert index
std::vector<std::vector<int>> expert_group_by_type(int num_types, std::vector<utils::Expert> experts)
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
void expert_sort_each_group_by_processtm(std::vector<std::vector<int>> &group_experts, std::vector<utils::Expert> experts)
{
    for (int i = 0; i < group_experts.size(); ++i)
    {
        std::sort(group_experts[i].begin(), group_experts[i].end(), [experts, i](int &a, int &b) -> bool {
            return experts[a].process_dura[i] < experts[b].process_dura[i];
        });
    }
}

int main(int argc, char const *argv[])
{
    std::vector<utils::Task> tasks = utils::load_work_order();
    std::vector<utils::Expert> experts = utils::load_expert_process_duras();
    int num_types = experts[0].num_types();
    std::vector<std::vector<utils::Task>> group_tasks = task_groupby_type(num_types, tasks);
    task_sort_each_group_by_tm_resptm(group_tasks);
    std::vector<std::vector<int>> group_experts = expert_group_by_type(num_types, experts);
    expert_sort_each_group_by_processtm(group_experts, experts);
    return 0;
}
