/**
 * Metrics for Task Assignment for Experts
 * This file contains metrics for judge the strategy
 * The metrics are: Balance of workload for each expert, response time for tasks and the execution efficiency
 */
#pragma once
#include "utils.hpp"
#include <algorithm>
#include <cmath>

namespace metrics
{
    double expert_workload(utils::Expert &expert)
    {
        return expert.busy_total_time / (60 * 8 * 3.0);
    }

    double task_response_timeout(utils::Task &task)
    {
        return std::max(task.start_process_tmpt - task.tm_stamp - task.max_resp_tm, 0) * 1.0 / task.max_resp_tm;
    }

    double task_exec_efficiency(utils::Task &task)
    {
        int total_stay = 0, size = task.each_stay_dura.size();
        for (int i = 0; i < size; ++i)
            total_stay += task.each_stay_dura[i];
        return total_stay * 1.0 / task.each_stay_dura[size - 1];
    }

    double score(std::vector<double> expert_workloads, std::vector<double> task_resp_tmout, std::vector<double> exec_effs)
    {
        double avg_eff = 0;
        for (int i = 0; i < exec_effs.size(); ++i)
            avg_eff += exec_effs[i];
        avg_eff /= exec_effs.size();
        double avg_tmout = 0;
        for (int i = 0; i < task_resp_tmout.size(); ++i)
            avg_tmout += task_resp_tmout[i];
        avg_tmout /= task_resp_tmout.size();
        double expert_avg_workload = 0;
        for (int i = 0; i < expert_workloads.size(); ++i)
            expert_avg_workload += expert_workloads[i];
        expert_avg_workload /= expert_workloads.size();
        double std_deriv = 0;
        for (int i = 0; i < expert_workloads.size(); ++i)
            std_deriv += pow(expert_workloads[i] - expert_avg_workload, 2);
        std_deriv = sqrt(std_deriv / expert_workloads.size());
        double score = 3000 * avg_eff / (2 * avg_tmout + 3 * std_deriv);
        return score;
    }

} // namespace metrics
