/**
 * This is a greedy method, based on best fit spt_benchmark, but add migration
 */
#include "monte_metrics.hpp"
#include "monte_utils.hpp"
#include "utils.hpp"
#include <omp.h>
#include <tuple>

static double EPSILON = 0.9;

/**
 * save result into csv file
 */
void save_result(std::vector<std::vector<int>> &result, double score)
{
    time_t date = time(nullptr);
    tm *date_tm = localtime(&date);
    char result_tm_stamp[50];
    sprintf(result_tm_stamp, "%02d%02d%02d_%02d%02d%02d_score_%lf.csv", date_tm->tm_year + 1900 - 2000,
            date_tm->tm_mon + 1, date_tm->tm_mday, date_tm->tm_hour, date_tm->tm_min, date_tm->tm_sec, score);
    char prefix[100] = "\0";
    strncpy(prefix, utils::PRED_RESULT_PREFIX, sizeof(utils::PRED_RESULT_PREFIX));
    utils::save_result(strcat(prefix, result_tm_stamp), result);
}

/**
 * Extract result, each array in the result is [task id, expert id , time]
 */
std::vector<std::vector<int>> extract_result(const std::vector<monte_utils::Task> &tasks, const std::vector<monte_utils::Expert> &experts)
{
    std::vector<std::vector<int>> result;
    for (int i = 0; i < tasks.size(); ++i)
    {
        for (int j = 0; j < tasks[i].curr_migrate_count; ++j)
        {
            result.emplace_back(std::vector<int>({tasks[i].task_id, experts[tasks[i].each_stay_expert_id[j]].expert_id, tasks[i].assign_tm[j]}));
        }
    }
    std::sort(result.begin(), result.end(), [](const std::vector<int> &a, const std::vector<int> &b) -> bool {
        if (a[2] != b[2])
            return a[2] < b[2];
        else
            return a[0] < b[0];
    });
    return result;
}

/**
 * check if two task swap is valid
 */
bool swap_check(monte_utils::Task &task_i, monte_utils::Task &task_j, monte_utils::Expert &expert_i, monte_utils::Expert &expert_j)
{
    bool flag1 = ((expert_i.process_type_duras[task_i.type] == monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (expert_i.process_type_duras[task_j.type] < monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (expert_j.process_type_duras[task_j.type] == monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (expert_j.process_type_duras[task_i.type] < monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (task_i.curr_migrate_count < monte_utils::TASK_MAX_MIGRATION) &&
                  (task_j.curr_migrate_count < monte_utils::TASK_MAX_MIGRATION)),
         flag2 = ((expert_i.process_type_duras[task_i.type] == monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (expert_i.process_type_duras[task_j.type] < monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (expert_j.process_type_duras[task_j.type] == monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (expert_j.process_type_duras[task_i.type] == monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (task_i.curr_migrate_count + 1 < monte_utils::TASK_MAX_MIGRATION) &&
                  (task_j.curr_migrate_count < monte_utils::TASK_MAX_MIGRATION)),
         flag3 = ((expert_i.process_type_duras[task_i.type] == monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (expert_i.process_type_duras[task_j.type] == monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (expert_j.process_type_duras[task_j.type] == monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (expert_j.process_type_duras[task_i.type] < monte_utils::EXPERT_NOT_GOOD_TIME) &&
                  (task_i.curr_migrate_count < monte_utils::TASK_MAX_MIGRATION) &&
                  (task_j.curr_migrate_count + 1 < monte_utils::TASK_MAX_MIGRATION));
    return flag1 || flag2 || flag3;
}

/**
 * Group experts by good at processing types, one expert may belong to multiple group
 */
std::vector<std::vector<int>> group_experts(const std::vector<monte_utils::Expert> &experts, const int num_types)
{
    std::vector<std::vector<int>> expt_groups(num_types);
    for (int i = 0; i < experts.size(); ++i)
    {
        for (int j = 0; j < num_types; ++j)
        {
            if (experts[i].process_type_duras[j] < monte_utils::EXPERT_NOT_GOOD_TIME)
                expt_groups[j].push_back(i);
        }
    }
    for (int i = 0; i < expt_groups.size(); ++i)
    {
        std::sort(expt_groups[i].begin(), expt_groups[i].end(), [&experts, i](const int a, const int b) -> bool {
            if (experts[a].process_type_duras[i] != experts[b].process_type_duras[i])
                return experts[a].process_type_duras[i] < experts[b].process_type_duras[i];
            else
                return experts[a].expert_id < experts[b].expert_id;
        });
    }
    return expt_groups;
}

/**
 * Assign a task to expert to process
 */
void assign_task(monte_utils::Task &task, monte_utils::Expert &expert, const int task_idx, const int expert_idx, const int env_tm)
{
    if (task.start_process_tm == -1)
        task.start_process_tm = env_tm;
    task.each_stay_expert_id[task.curr_migrate_count] = expert_idx;
    task.assign_tm[task.curr_migrate_count] = env_tm;
    task.curr_migrate_count++;
    for (int i = 0; i < monte_utils::EXPERT_MAX_PARALLEL; ++i)
    {
        if (expert.channels[i] == -1)
        {
            expert.channels[i] = task_idx;
            expert.num_idle_channel--;
            break;
        }
    }
}

/**
 * Release task from expert
 */
void release_task(monte_utils::Expert &expert, const int task_idx)
{
    for (int i = 0; i < monte_utils::EXPERT_MAX_PARALLEL; ++i)
    {
        if (expert.channels[i] == task_idx)
        {
            expert.channels[i] = -1;
            expert.num_idle_channel++;
            break;
        }
    }
}

/**
 * swap two tasks
 */
void swap_tasks(monte_utils::Task &task_a, monte_utils::Task &task_b,
                monte_utils::Expert &expert_a, monte_utils::Expert &expert_b, int task_a_idx, int task_b_idx,
                int expert_a_idx, int expert_b_idx, int env_tm)
{
    // release resources
    release_task(expert_a, task_a_idx);
    release_task(expert_b, task_b_idx);
    assign_task(task_a, expert_b, task_a_idx, expert_b_idx, env_tm);
    assign_task(task_b, expert_a, task_b_idx, expert_a_idx, env_tm);
}

/**
 * Best fit add migrations
 */
std::tuple<std::vector<std::vector<int>>, double> run_alg(std::vector<monte_utils::Task> tasks,
                                                          std::vector<monte_utils::Expert> experts,
                                                          std::vector<std::vector<int>> &expt_groups)
{
    int num_finish = 0, env_tm = 0;
    while (num_finish < tasks.size())
    {
        std::vector<bool> vis(tasks.size(), false);
        // check finish
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (tasks[i].curr_migrate_count == 0 || tasks[i].finish_tm != -1)
                continue;
            int expert_idx = tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1];
            int process_due = tasks[i].assign_tm[tasks[i].curr_migrate_count - 1] + experts[expert_idx].process_type_duras[tasks[i].type];
            if (env_tm == process_due)
            {
                release_task(experts[expert_idx], i);
                tasks[i].finish_tm = env_tm;
                vis[i] = true;
                num_finish++;
            }
        }
        // check new generated tasks, try assign to best fit expert
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (tasks[i].curr_migrate_count > 0 || vis[i])
                continue;
            else if (tasks[i].generate_tm > env_tm)
                break;
            int task_type = tasks[i].type;
            for (int &expt_idx : expt_groups[task_type])
            {
                double rand_val = (double)rand() / RAND_MAX;
                if (experts[expt_idx].num_idle_channel > 0 && rand_val < EPSILON)
                {
                    assign_task(tasks[i], experts[expt_idx], i, expt_idx, env_tm);
                    std::sort(expt_groups[task_type].begin(), expt_groups[task_type].end(), [&experts, task_type](const int a, const int b) -> bool {
                        if (experts[a].process_type_duras[task_type] != experts[b].process_type_duras[task_type])
                            return experts[a].process_type_duras[task_type] < experts[b].process_type_duras[task_type];
                        else if (experts[a].busy_sum != experts[b].busy_sum)
                            return experts[a].busy_sum < experts[b].busy_sum;
                        else if (experts[a].num_idle_channel != experts[b].num_idle_channel)
                            return experts[a].num_idle_channel > experts[b].num_idle_channel;
                        else
                            return experts[a].expert_id < experts[b].expert_id;
                    });
                    vis[i] = true;
                    break;
                }
            }
        }
        // check for new generated tasks that not found best fit
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (vis[i] || tasks[i].curr_migrate_count > 0)
                continue;
            else if (tasks[i].generate_tm > env_tm)
                break;
            for (int j = experts.size() - 1; j >= 0; --j)
            {
                double rand_val = (double)rand() / RAND_MAX;
                if (experts[j].num_idle_channel > 0 && rand_val < RAND_MAX)
                {
                    assign_task(tasks[i], experts[j], i, j, env_tm);
                    vis[i] = true;
                    break;
                }
            }
        }
        // check for migrate for already assigned tasks
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (vis[i] || tasks[i].finish_tm != -1 || tasks[i].curr_migrate_count == 0 ||
                tasks[i].curr_migrate_count == monte_utils::TASK_MAX_MIGRATION ||
                experts[tasks[i]
                            .each_stay_expert_id[tasks[i].curr_migrate_count - 1]]
                        .process_type_duras[tasks[i].type] < monte_utils::EXPERT_NOT_GOOD_TIME)
                continue;
            else if (tasks[i].generate_tm > env_tm)
                break;
            int task_type = tasks[i].type;
            for (int &expt_idx : expt_groups[task_type])
            {
                double rand_val = (double)rand() / RAND_MAX;
                if (experts[expt_idx].num_idle_channel > 0 && rand_val < EPSILON)
                {
                    assign_task(tasks[i], experts[expt_idx], i, expt_idx, env_tm);
                    std::sort(expt_groups[task_type].begin(), expt_groups[task_type].end(), [&experts, task_type](const int a, const int b) -> bool {
                        if (experts[a].process_type_duras[task_type] != experts[b].process_type_duras[task_type])
                            return experts[a].process_type_duras[task_type] < experts[b].process_type_duras[task_type];
                        else if (experts[a].busy_sum != experts[b].busy_sum)
                            return experts[a].busy_sum < experts[b].busy_sum;
                        else if (experts[a].num_idle_channel != experts[b].num_idle_channel)
                            return experts[a].num_idle_channel > experts[b].num_idle_channel;
                        else
                            return experts[a].expert_id < experts[b].expert_id;
                    });
                    vis[i] = true;
                    break;
                }
            }
        }
        // check swap
        for (int i = 0; i < tasks.size() - 1 && !vis[i]; ++i)
        {
            if (vis[i] || tasks[i].finish_tm != -1 || tasks[i].curr_migrate_count == 0 ||
                tasks[i].curr_migrate_count == monte_utils::TASK_MAX_MIGRATION)
                continue;
            int expt_i_idx = tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1];
            for (int j = i + 1; j < tasks.size() && !vis[j]; ++j)
            {
                if (vis[j] || tasks[j].finish_tm != -1 || tasks[j].curr_migrate_count == 0 ||
                    tasks[j].curr_migrate_count == monte_utils::TASK_MAX_MIGRATION)
                    continue;
                int expt_j_idx = tasks[j].each_stay_expert_id[tasks[j].curr_migrate_count - 1];
                double rand_val = (double)rand() / RAND_MAX;
                if (swap_check(tasks[i], tasks[j], experts[expt_i_idx], experts[expt_j_idx]) && rand_val < EPSILON)
                {
                    swap_tasks(tasks[i], tasks[j], experts[expt_i_idx], experts[expt_j_idx], i, j, expt_i_idx, expt_j_idx, env_tm);
                    vis[i] = true;
                    vis[j] = true;
                    break;
                }
            }
        }

        env_tm++;
        for (monte_utils::Expert &expt : experts)
        {
            if (expt.num_idle_channel < monte_utils::EXPERT_MAX_PARALLEL)
                expt.busy_sum++;
        }
    }

    std::vector<std::vector<int>> solution = extract_result(tasks, experts);
    double score = monte_metrics::score(tasks, experts);
    return std::make_tuple(solution, score);
}

int main(int argc, char const *argv[])
{
    srand(time(nullptr));
    for (int iter = 1; iter <= 1000; ++iter)
    {
        printf("Iter #%05d ...\n", iter);
        std::vector<monte_utils::Task> init_tasks = monte_utils::load_tasks();
        std::vector<monte_utils::Expert> init_experts = monte_utils::load_experts();
        std::vector<std::vector<monte_utils::Task>> thread_tasks(8);
        std::vector<std::vector<monte_utils::Expert>> thread_experts(8);
        for (int i = 0; i < 8; ++i)
        {
            thread_tasks[i] = init_tasks;
            thread_experts[i] = init_experts;
        }
#pragma omp parallel for
        for (int pa = 1; pa <= 8; ++pa)
        {
            std::vector<monte_utils::Task> tasks = thread_tasks[pa - 1];
            std::vector<monte_utils::Expert> experts = thread_experts[pa - 1];
            std::vector<std::vector<int>> expt_grps = group_experts(experts, monte_utils::NUM_TASK_TYPE);
            std::sort(tasks.begin(), tasks.end(), [](const monte_utils::Task &a, const monte_utils::Task &b) -> bool {
                if (a.generate_tm != b.generate_tm)
                    return a.generate_tm < b.generate_tm;
                else if (a.max_resp != b.max_resp)
                    return a.max_resp < b.max_resp;
                else
                    return a.task_id < b.task_id;
            });
            std::tuple<std::vector<std::vector<int>>, double> ret = run_alg(tasks, experts, expt_grps);
            save_result(std::get<0>(ret), std::get<1>(ret));
        }

        if (iter % 100 == 0)
            iter *= 0.9;
    }
    return 0;
}
