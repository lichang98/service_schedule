/**
 * This file contains methods for calculating metrics
 */

#pragma once
#include "monte_utils.hpp"
#include <algorithm>
#include <cmath>

namespace monte_metrics
{

double expert_workload(monte_utils::Expert &expert)
{
    return expert.busy_sum / (60 * 8 * 3.0);
}

double task_timeout(monte_utils::Task &task)
{
    return std::max(task.start_process_tm - task.generate_tm - task.max_resp, 0) * 1.0 / task.max_resp;
}

double task_exec_eff(monte_utils::Task &task)
{
    int last_stay = task.finish_tm - task.assign_tm[task.curr_migrate_count - 1];
    return last_stay * 1.0 / (task.finish_tm - task.generate_tm);
}

double score(std::vector<monte_utils::Task> &tasks, std::vector<monte_utils::Expert> &experts)
{
    double avg_exec_eff = 0, avg_timeout = 0;
    for (int i = 0; i < tasks.size(); ++i)
    {
        avg_exec_eff += task_exec_eff(tasks[i]);
        avg_timeout += task_timeout(tasks[i]);
    }
    avg_exec_eff /= tasks.size();
    avg_timeout /= tasks.size();

    double workload_std = 0, avg_workload = 0;
    for (int i = 0; i < experts.size(); ++i)
        avg_workload += expert_workload(experts[i]);
    avg_workload /= experts.size();
    for (int i = 0; i < experts.size(); ++i)
    {
        double val = expert_workload(experts[i]);
        workload_std += (val - avg_workload) * (val - avg_workload);
    }
    workload_std = sqrt(workload_std / ((int)experts.size()));
    return 3000 * avg_exec_eff / (3 * avg_timeout + 2 * workload_std);
}

} // namespace monte_metrics
