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
    return std::max(task.start_process_tm - task.generate_tm - task.max_resp, 0);
}

double task_exec_eff(monte_utils::Task &task)
{
    if (task.curr_migrate_count == 0)
        return 0;
    int total_stay = 0;
    for (int i = 0; i < task.curr_migrate_count - 1; ++i)
        total_stay += task.assign_tm[i + 1] - task.assign_tm[i];
    total_stay += task.finish_tm - task.assign_tm[task.curr_migrate_count - 1];
    return (task.finish_tm - task.assign_tm[task.curr_migrate_count - 1]) / total_stay * 1.0;
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
    avg_workload /= (int)experts.size();
    for (int i = 0; i < experts.size(); ++i)
    {
        double val = expert_workload(experts[i]);
        workload_std += (val - avg_workload) * (val - avg_workload);
    }
    workload_std = sqrt(workload_std / (experts.size() - 1));
    printf("avg_exec_eff=%lf, avg_timeout=%lf, workload_std=%lf\n", avg_exec_eff, avg_timeout, workload_std);
    return 3000 * avg_exec_eff / (2 * avg_timeout + 3 * workload_std);
}

} // namespace monte_metrics
