/**
 * This file contains GA method
 * Each solution contains all tasks actions, each task's action is
 *  represented as a array with 5 integers. In the array, the value
 *  represented the index of the expert, -1 represent no expert assigned.
 */
#include "monte_metrics.hpp"
#include "monte_utils.hpp"
#include "utils.hpp"
#include <cstring>
#include <ctime>
#include <set>
#include <tuple>
#include <unistd.h>

#define RANDOM(low, high) ((int)(rand() % (high - low + 1) + low))

static const int NUM_INIT_SOLUTIONS = 100; // the initial generated ga solutions
static const int NUM_MUTATIONS = 10;
static const double MUTATION_RATIO = 0.1; // the ratio of the tasks that actions will be changed
static const int NUM_ITERS = 1000;

/**
 * Group experts by good at processing types, one expert may belong to multiple group
 */
std::vector<std::vector<int>> group_experts(std::vector<monte_utils::Expert> &experts, int num_types)
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
 * Detect dependency cycle
 * link matrix representation of graph
 * pos (i,j) value k means there are k tasks try to migrate from expert i to expert j
 */
bool detect_dep_cycle(std::vector<std::vector<int>> &link_mat)
{
    int row = link_mat.size(), col = link_mat[0].size();
    bool flag = true;
    while (flag)
    {
        flag = false;
        for (int i = 0; i < row; ++i)
        {
            bool zero_out = true;
            for (int j = 0; j < col && zero_out; ++j)
            {
                if (link_mat[i][j] > 0)
                    zero_out = false;
            }
            if (zero_out)
            {
                flag = true;
                // remove coresponding column
                for (int ln = 0; ln < row; ++ln)
                {
                    if (link_mat[ln][i] > 0)
                        link_mat[ln][i]=0;
                }
            }
        }
    }
    flag = false;
    for (int i = 0; i < row && !flag; ++i)
        for (int j = 0; j < col && !flag; ++j)
            if (link_mat[i][j] > 0)
                flag = true;
    return flag;
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
    int env_tm = 0, num_finish = 0;
    std::vector<int> start_poses(tasks.size(), 0);
    std::vector<bool> finish_flag(tasks.size(), false);
    for (int i = 0; i < tasks.size(); ++i)
    {
        int start_pos = 0;
        while (s[i * monte_utils::TASK_MAX_MIGRATION + start_pos] == -1)
            start_pos++;
        start_poses[i] = start_pos;
    }
    while (num_finish < tasks.size())
    {

        // detect deps cycle , test
        std::vector<std::vector<int>> link_mat(experts.size(), std::vector<int>(experts.size(), 0));
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (finish_flag[i] || start_poses[i] == monte_utils::TASK_MAX_MIGRATION || tasks[i].curr_migrate_count == 0)
                continue;
            int prev_expt_idx = tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1],
                next_expt_idx = s[i * monte_utils::TASK_MAX_MIGRATION + start_poses[i]];
            link_mat[prev_expt_idx][next_expt_idx]++;
        }
        std::cout << "start detection.." << std::endl;
        if (detect_dep_cycle(link_mat))
        {
            std::cout << "detected cycles.....!!!!" << std::endl;
            sleep(1);
        }
        else
        {
            std::cout << "no cycle" << std::endl;
        }
        

        // firstly check if tasks have finished on current expert
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (tasks[i].curr_migrate_count == 0)
                continue;
            int process_tm = experts[tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1]].process_type_duras[tasks[i].type];
            if (env_tm == tasks[i].assign_tm[tasks[i].curr_migrate_count - 1] + process_tm + 1)
            {
                tasks[i].finish_tm = env_tm;
                finish_flag[i] = true;
                num_finish++;
                // release channel of expert
                int expt_idx = tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1];
                for (int j = 0; j < monte_utils::EXPERT_MAX_PARALLEL; ++j)
                {
                    if (experts[expt_idx].channels[j] == i)
                    {
                        experts[expt_idx].channels[j] = -1;
                        experts[expt_idx].num_idle_channel++;
                        break;
                    }
                }
            }
        }

        // check migration
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (finish_flag[i] || start_poses[i] == monte_utils::TASK_MAX_MIGRATION || tasks[i].generate_tm > env_tm)
                continue;
            int next_expt_idx = s[i * monte_utils::TASK_MAX_MIGRATION + start_poses[i]];
            if (experts[next_expt_idx].num_idle_channel > 0)
            {
                // release from previous expert
                if (tasks[i].curr_migrate_count > 0)
                {
                    int prev_expt_idx = tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1];
                    for (int j = 0; j < monte_utils::EXPERT_MAX_PARALLEL; ++j)
                    {
                        if (experts[prev_expt_idx].channels[j] == i)
                        {
                            experts[prev_expt_idx].channels[j] = -1;
                            experts[prev_expt_idx].num_idle_channel++;
                            break;
                        }
                    }
                }
                // assign to expert
                if (tasks[i].curr_migrate_count == 0)
                    tasks[i].start_process_tm = env_tm;
                tasks[i].assign_tm[tasks[i].curr_migrate_count] = env_tm;
                tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count] = next_expt_idx;
                tasks[i].curr_migrate_count++;
                start_poses[i]++;
                for (int j = 0; j < monte_utils::EXPERT_MAX_PARALLEL; ++j)
                {
                    if (experts[next_expt_idx].channels[j] == -1)
                    {
                        experts[next_expt_idx].channels[j] = i;
                        experts[next_expt_idx].num_idle_channel--;
                        break;
                    }
                }
            }
        }

        for (int i = 0; i < experts.size(); ++i)
        {
            if (experts[i].num_idle_channel < monte_utils::EXPERT_MAX_PARALLEL)
                experts[i].busy_sum++;
        }
        env_tm++;
        if (env_tm % 10 == 0)
            std::cout << "\t\ttime=" << env_tm << ", num finish=" << num_finish << std::endl;
    }
    std::cout << std::endl;
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
        for (int i = 0; i < solutions.size(); ++i)
            result_scores.emplace_back(convert_solution_to_result(solutions[i], tasks, experts));
        std::cout << "\tsolutions simulate finish.." << std::endl;
        double max_score = 0, min_score = 0x7fffffff;
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
        std::cout << "\tbest score=" << max_score << std::endl;
    }
    return best_result;
}

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

int main(int argc, char const *argv[])
{
    srand(time(NULL));
    std::vector<monte_utils::Task> tasks = monte_utils::load_tasks();
    std::vector<monte_utils::Expert> experts = monte_utils::load_experts();
    std::vector<std::vector<int>> expt_groups = group_experts(experts, monte_utils::NUM_TASK_TYPE);
    std::vector<std::vector<int>> result = ga_run(tasks, experts, expt_groups);
    save_result(result);
    return 0;
}
