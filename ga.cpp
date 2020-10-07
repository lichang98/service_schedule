/**
 * This file contains GA method
 * Each solution contains all tasks actions, each task's action is
 *  represented as a array with 5 integers. In the array, the value
 *  represented the index of the expert, -1 represent no expert assigned.
 */
#include "monte_metrics.hpp"
#include "monte_utils.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cstring>
#include <ctime>
#include <omp.h>
#include <set>
#include <tuple>

#define RANDOM(low, high) ((int)(rand() % (high - low + 1) + low))

static const int NUM_INIT_SOLUTIONS = 128; // the initial generated ga solutions
static const int NUM_MUTATIONS = 10;
static const double MUTATION_RATIO = 0.1; // the ratio of the tasks that actions will be changed
static const int NUM_ITERS = 1000;
static const int MAX_TIME_LONG = 1000000;

/**
 * save result into csv file
 */
void save_result(std::vector<std::vector<int>> &result)
{
    time_t date = time(nullptr);
    tm *date_tm = localtime(&date);
    char result_tm_stamp[50];
    sprintf(result_tm_stamp, "%02d%02d%02d_%02d%02d%02d.csv", date_tm->tm_year + 1900 - 2000,
            date_tm->tm_mon + 1, date_tm->tm_mday, date_tm->tm_hour, date_tm->tm_min, date_tm->tm_sec);
    utils::save_result(strcat(utils::PRED_RESULT_PREFIX, result_tm_stamp), result);
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
 * Initial generate solutions for GA
 * each solution is an array with size num tasks * 5
 * for each task, there are 5 integers, such as [-1,-1,3,4,12] which means
 * that the task migrate via expert index with 3,4 and 12.
 * what shoule be empahsised is that the last expert should good at processing
 * the task and all previous experts all should be not good at processing the task
 */
std::vector<std::vector<int>> ga_init_solutions(std::vector<monte_utils::Task> &tasks,
                                                std::vector<std::vector<int>> &expt_groups)
{
    std::vector<std::vector<int>> solutions(NUM_INIT_SOLUTIONS, std::vector<int>(monte_utils::TASK_MAX_MIGRATION * tasks.size(), -1));
    for (int i = 0; i < NUM_INIT_SOLUTIONS; ++i)
    {
        for (int j = 0; j < tasks.size(); ++j)
        {
            int start_pos = RANDOM(j * monte_utils::TASK_MAX_MIGRATION, (j + 1) * monte_utils::TASK_MAX_MIGRATION - 1);
            int prev_expert_idx = -1;
            while (start_pos < (j + 1) * monte_utils::TASK_MAX_MIGRATION - 1)
            {
                // set not suitable experts idx
                int rand_group = RANDOM(0, expt_groups.size() - 1);
                int expert_idx = expt_groups[rand_group][RANDOM(0, expt_groups[rand_group].size() - 1)];
                // make sure that the consecutive two experts are not same
                while (expert_idx == prev_expert_idx)
                {
                    rand_group = RANDOM(0, expt_groups.size() - 1);
                    expert_idx = expt_groups[rand_group][RANDOM(0, expt_groups[rand_group].size() - 1)];
                }
                prev_expert_idx = expert_idx;
                solutions[i][start_pos] = expert_idx;
                start_pos++;
            }
            // the last expert must be suitable
            int group_idx = tasks[j].type;
            solutions[i][start_pos] = expt_groups[group_idx][RANDOM(0, expt_groups[group_idx].size() - 1)];
        }
    }
    return solutions;
}

/**
 * The crossover operation of two solution
 */
std::vector<int> crossover(std::vector<int> &s1, std::vector<int> &s2)
{
    std::vector<int> s_n = std::vector<int>(s1.size(), -1);
    for (int i = 0; i < s1.size(); i += 2 * monte_utils::TASK_MAX_MIGRATION)
    {
        for (int j = i; j < i + monte_utils::TASK_MAX_MIGRATION; ++j)
            s_n[j] = s1[j];
    }
    for (int i = monte_utils::TASK_MAX_MIGRATION; i < s2.size(); i += 2 * monte_utils::TASK_MAX_MIGRATION)
    {
        for (int j = i; j < i + monte_utils::TASK_MAX_MIGRATION; ++j)
            s_n[j] = s2[j];
    }
    return s_n;
}

/**
 * The actions for part of the tasks will be changed
 */
void mutation(std::vector<int> &s, std::vector<monte_utils::Task> &tasks, std::vector<std::vector<int>> &expt_groups)
{
    std::set<int> mut_idxs;
    int task_count = (int)s.size() / monte_utils::TASK_MAX_MIGRATION;
    int mut_count = (int)(MUTATION_RATIO * task_count);
    while (mut_idxs.size() < mut_count)
        mut_idxs.insert(RANDOM(0, task_count - 1));
    // mutate
    for (const int &idx : mut_idxs)
    {
        int start_pos = idx * monte_utils::TASK_MAX_MIGRATION;
        start_pos = RANDOM(start_pos, start_pos + monte_utils::TASK_MAX_MIGRATION - 1);
        for (int i = idx * monte_utils::TASK_MAX_MIGRATION; i < start_pos; ++i)
            s[i] = -1;
        int prev_expert_idx = -1;
        while (start_pos < (idx + 1) * monte_utils::TASK_MAX_MIGRATION - 1)
        {
            // set not suitable experts idx
            int rand_group = RANDOM(0, expt_groups.size() - 1);
            int curr_expert_idx = expt_groups[rand_group][RANDOM(0, expt_groups[rand_group].size() - 1)];
            while (curr_expert_idx == prev_expert_idx)
            {
                rand_group = RANDOM(0, expt_groups.size() - 1);
                curr_expert_idx = expt_groups[rand_group][RANDOM(0, expt_groups[rand_group].size() - 1)];
            }
            prev_expert_idx = curr_expert_idx;
            s[start_pos] = curr_expert_idx;
            start_pos++;
        }
        // last expert must be suitable
        s[start_pos] = expt_groups[tasks[idx].type][RANDOM(0, expt_groups[tasks[idx].type].size() - 1)];
    }
}

/**
 * Extract result, each array in the result is [task id, expert id , time]
 */
std::vector<std::vector<int>> extract_result(std::vector<monte_utils::Task> &tasks)
{
    std::vector<std::vector<int>> result;
    for (int i = 0; i < tasks.size(); ++i)
    {
        for (int j = 0; j < tasks[i].curr_migrate_count; ++j)
        {
            result.emplace_back(std::vector<int>({tasks[i].task_id, tasks[i].each_stay_expert_id[j], tasks[i].assign_tm[j]}));
        }
    }
    return result;
}

/**
 * check interval of vector all positive
 */
bool check_interval(std::vector<int> &vec, int low, int high)
{
    for (int i = low; i < high; ++i)
    {
        if (vec[i] <= 0)
            return false;
    }
    return true;
}

/**
 * convert solution into result
 * The function will simulate according to the solution, tasks and experts's record variables will be changed
 * the result is formed with array of [task id , expert id, time]
 * @return the tuple of result format and score
 */
std::tuple<std::vector<std::vector<int>>, double> convert_solution_to_result(std::vector<int> &s,
                                                                             std::vector<monte_utils::Task> tasks,
                                                                             std::vector<monte_utils::Expert> experts)
{
    std::vector<std::vector<int>> expert_marker(experts.size());
    for (int i = 0; i < experts.size(); ++i)
        expert_marker[i] = std::vector<int>(MAX_TIME_LONG, monte_utils::EXPERT_MAX_PARALLEL);
    for (int i = 0; i < tasks.size(); ++i)
    {
        int start_pos = i * monte_utils::TASK_MAX_MIGRATION, base = i * monte_utils::TASK_MAX_MIGRATION;
        while (s[start_pos] == -1)
            start_pos++;
        tasks[i].curr_migrate_count = monte_utils::TASK_MAX_MIGRATION - (start_pos - base); // the total migration count
        for (int j = 0; j < tasks[i].curr_migrate_count; ++j)
        {
            tasks[i].each_stay_expert_id[j] = s[start_pos + j];
            tasks[i].assign_tm[j] = tasks[i].generate_tm + j;
        }
        // int final_process_tm = experts[tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1]].process_type_duras[tasks[i].type];
        std::vector<int> process_times(tasks[i].curr_migrate_count);
        int task_type = tasks[i].type;
        for (int j = 0; j < tasks[i].curr_migrate_count; ++j)
            process_times[j] = experts[tasks[i].each_stay_expert_id[j]].process_type_duras[task_type];
        int migrate_count = tasks[i].curr_migrate_count;
        // check valid intervals
        bool flag = false;
        while (!flag)
        {
            flag = true;
            std::vector<int> start_times(migrate_count + 1, 0);
            for (int j = 0; j < migrate_count; ++j)
                start_times[j] = tasks[i].assign_tm[j];
            start_times[migrate_count] = start_times[migrate_count - 1] + process_times[migrate_count - 1];
            for (int j = 0; j < migrate_count; ++j)
            {
                if (!check_interval(expert_marker[tasks[i].each_stay_expert_id[j]], start_times[j], start_times[j + 1]))
                {
                    flag = false;
                    tasks[i].assign_tm[j]++;
                    for (int k = j + 1; k < tasks[i].curr_migrate_count; ++k)
                        tasks[i].assign_tm[k] = std::max(tasks[i].assign_tm[k - 1] + 1, tasks[i].assign_tm[k]);
                    break;
                }
            }
        }
        // fill time intervales
        // the tasks may finish at intermediate expert, need to check
        std::vector<int> start_times(migrate_count + 1, 0);
        for (int j = 0; j < migrate_count; ++j)
            start_times[j] = tasks[i].assign_tm[j];
        start_times[migrate_count] = start_times[migrate_count - 1] + process_times[migrate_count - 1];
        for (int j = 0; j < migrate_count; ++j)
        {
            if (start_times[j + 1] - start_times[j] >= process_times[j])
            {
                // task finish on the expert
                if (j != migrate_count - 1)
                {
                    std::fill(tasks[i].each_stay_expert_id + j + 1, tasks[i].each_stay_expert_id + migrate_count, -1);
                    std::fill(tasks[i].assign_tm + j + 1, tasks[i].assign_tm + migrate_count, -1);
                    tasks[i].curr_migrate_count = j + 1;
                }
                for (int k = start_times[j]; k < start_times[j] + process_times[j]; ++k)
                    expert_marker[tasks[i].each_stay_expert_id[j]][k] -= 1;
                break;
            }
            else
            {
                for (int k = start_times[j]; k < start_times[j + 1]; ++k)
                    expert_marker[tasks[i].each_stay_expert_id[j]][k] -= 1;
            }
        }
    }
    // update experts
    for (int i = 0; i < experts.size(); ++i)
    {
        for (int j = 0; j < MAX_TIME_LONG; ++j)
        {
            if (expert_marker[i][j] < monte_utils::EXPERT_MAX_PARALLEL)
                experts[i].busy_sum++;
        }
    }
    std::vector<std::vector<int>> result = extract_result(tasks);
    double score = monte_metrics::score(tasks, experts);
    return std::make_tuple(result, score);
}

/**
 * Run GA algorithm
 */
std::vector<std::vector<int>> ga_run(std::vector<monte_utils::Task> &tasks,
                                     std::vector<monte_utils::Expert> &experts, std::vector<std::vector<int>> &expt_groups)
{
    std::cout << "Initial solutions..." << std::endl;
    std::vector<std::vector<int>> solutions = ga_init_solutions(tasks, expt_groups);
    std::vector<std::vector<int>> best_result;
    std::cout << "Start GA method...." << std::endl;
    for (int iter = 1; iter <= NUM_ITERS; ++iter)
    {
        std::vector<std::tuple<std::vector<std::vector<int>>, double>> result_scores;
        std::cout << "Iter #" << iter << ": start simulations for solutions..." << std::endl;
        result_scores.resize(solutions.size());
#pragma omp parallel for
        for (int i = 0; i < solutions.size(); ++i)
            result_scores[i] = convert_solution_to_result(solutions[i], tasks, experts);
        std::cout << "\tsolutions simulate finish.." << std::endl;
        double max_score = 0, min_score = 1e8;
        int max_score_idx = 0, min_score_idx = 0;
        for (int i = 0; i < result_scores.size(); ++i)
        {
            if (std::get<1>(result_scores[i]) > max_score)
            {
                max_score = std::get<1>(result_scores[i]);
                max_score_idx = i;
            }
            if (std::get<1>(result_scores[i]) < min_score)
            {
                min_score = std::get<1>(result_scores[i]);
                min_score_idx = i;
            }
        }
        best_result = std::get<0>(result_scores[max_score_idx]);
        save_result(best_result);
        int s2 = RANDOM(0, result_scores.size() - 1);
        while (s2 == max_score_idx || s2 == min_score_idx)
            s2 = RANDOM(0, result_scores.size() - 1);
        // crossover
        std::vector<int> s_nw = crossover(solutions[max_score_idx], solutions[s2]);
        solutions.emplace_back(s_nw);
        // mutations
        for (int i = 0; i < NUM_MUTATIONS; ++i)
        {
            int idx = RANDOM(0, solutions.size() - 1);
            while (idx == max_score_idx)
                idx = RANDOM(0, solutions.size() - 1);
            mutation(solutions[idx], tasks, expt_groups);
        }
        solutions.erase(solutions.begin() + min_score_idx);
        std::cout << "\tbest score=" << max_score << ", min score=" << min_score << std::endl;
    }
    return best_result;
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));
    std::vector<monte_utils::Task> tasks = monte_utils::load_tasks();
    std::vector<monte_utils::Expert> experts = monte_utils::load_experts();
    std::vector<std::vector<int>> expt_groups = group_experts(experts, monte_utils::NUM_TASK_TYPE);
    std::vector<std::vector<int>> result = ga_run(tasks, experts, expt_groups);
    return 0;
}
