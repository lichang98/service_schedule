/**
 * This file contains method of greedly dispatch tasks
 */
#include "monte_metrics.hpp"
#include "monte_utils.hpp"
#include "utils.hpp"
#include <ctime>
#include <omp.h>
#include <set>
#include <tuple>

static double EPSILON = 0.9;

struct SnapShot
{
    std::vector<monte_utils::Task> tasks;
    std::vector<monte_utils::Expert> experts;
    std::vector<bool> flags_finish;
    int snap_shot_tm;
    int score; // the score of the scheme which the snapshot belongs to

    SnapShot() : snap_shot_tm(0), score(0) {}
    SnapShot(int _snap_shot_tm, std::vector<monte_utils::Task> _tasks, std::vector<monte_utils::Expert> _experts, std::vector<bool> _flag_finish)
    {
        this->snap_shot_tm = _snap_shot_tm;
        this->score = 0;
        tasks.resize(_tasks.size());
        for (int i = 0; i < tasks.size(); ++i)
            tasks[i] = _tasks[i];
        experts.resize(_experts.size());
        for (int i = 0; i < experts.size(); ++i)
            experts[i] = _experts[i];
        flags_finish.resize(_flag_finish.size());
        for (int i = 0; i < _flag_finish.size(); ++i)
            flags_finish[i] = _flag_finish[i];
    }
};

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
 * check if two task swap is valid
 */
bool swap_check(monte_utils::Task &task_i, monte_utils::Task &task_j, monte_utils::Expert &expert_i, monte_utils::Expert &expert_j)
{
    bool flag = (expert_i.process_type_duras[task_i.type] == monte_utils::EXPERT_NOT_GOOD_TIME && expert_i.process_type_duras[task_j.type] < monte_utils::EXPERT_NOT_GOOD_TIME && expert_j.process_type_duras[task_j.type] == monte_utils::EXPERT_NOT_GOOD_TIME && expert_j.process_type_duras[task_i.type] < monte_utils::EXPERT_NOT_GOOD_TIME && task_i.curr_migrate_count < monte_utils::TASK_MAX_MIGRATION && task_j.curr_migrate_count < monte_utils::TASK_MAX_MIGRATION) ||
                (expert_i.process_type_duras[task_i.type] == monte_utils::EXPERT_NOT_GOOD_TIME && expert_i.process_type_duras[task_j.type] < monte_utils::EXPERT_NOT_GOOD_TIME && expert_j.process_type_duras[task_j.type] == monte_utils::EXPERT_NOT_GOOD_TIME && expert_j.process_type_duras[task_i.type] == monte_utils::EXPERT_NOT_GOOD_TIME && task_i.curr_migrate_count + 1 < monte_utils::TASK_MAX_MIGRATION && task_j.curr_migrate_count < monte_utils::TASK_MAX_MIGRATION) ||
                (expert_i.process_type_duras[task_i.type] == monte_utils::EXPERT_NOT_GOOD_TIME && expert_i.process_type_duras[task_j.type] == monte_utils::EXPERT_NOT_GOOD_TIME && expert_j.process_type_duras[task_j.type] == monte_utils::EXPERT_NOT_GOOD_TIME && expert_j.process_type_duras[task_i.type] < monte_utils::EXPERT_NOT_GOOD_TIME && task_i.curr_migrate_count < monte_utils::TASK_MAX_MIGRATION && task_j.curr_migrate_count + 1 < monte_utils::TASK_MAX_MIGRATION);
    return flag;
}

/**
 * The algorithm will run based on a policy
 * tasks will try to assign to suitable expert if available, if not assign to not suitable experts to reduce timeout
 * during the algorithm iteration, firstly check if have tasks finish, then check if tasks generated need to assign
 * and then try swap tasks in experts if oneside is suitable for another,and at this step, max migration restriction 
 * need to be checked, must keep sure that the task must finally been executed on suitable expert
 * @param snapshot_env_tm: if use snapshot, the simulation will start from the snapshot status
 * @param use_snapshot: the flag of is start from snap shot status
 * @param take_snapshot_gap: the time duration between two snapshots
 * @return return the convert result format, and random snap shots list
 */
std::tuple<std::vector<std::vector<int>>, double, std::vector<SnapShot>> run_alg(std::vector<monte_utils::Task> tasks, std::vector<monte_utils::Expert> experts,
                                                                                 std::vector<bool> flag_fin,
                                                                                 const std::vector<std::vector<int>> &expt_groups,
                                                                                 const int snapshot_env_tm, bool use_snapshot, const int take_snapshot_gap = 50)
{
    int env_tm = 0, num_finish = 0;
    if (use_snapshot)
        env_tm = snapshot_env_tm;
    std::vector<SnapShot> snapshots;
    int snap_shot_beg = (int)((double)rand() / RAND_MAX * 200); // random choose snapshot start time
    std::vector<bool> flags_finish;
    if (use_snapshot)
    {
        flags_finish = flag_fin;
        for (int i = 0; i < flags_finish.size(); ++i)
            if (flags_finish[i])
                num_finish++;
    }
    else
    {
        flags_finish = std::vector<bool>(tasks.size(), false);
    }
    while (num_finish < tasks.size())
    {
        std::vector<bool> flags_vis(tasks.size(), false);
        // check if tasks finish
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (!(flags_finish[i] || tasks[i].curr_migrate_count == 0))
            {
                int process_dura = experts[tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1]].process_type_duras[tasks[i].type];
                if (tasks[i].assign_tm[tasks[i].curr_migrate_count - 1] + process_dura + 1 == env_tm)
                {
                    flags_finish[i] = true;
                    num_finish++;
                    tasks[i].finish_tm = env_tm;
                    // release expert resource
                    int expt_idx = tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1];
                    release_task(experts[expt_idx], i);
                    flags_vis[i] = true;
                }
            }
        }

        // check if have generated tasks
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (tasks[i].generate_tm > env_tm || tasks[i].curr_migrate_count > 0)
                continue;
            bool flag_suc = false;
            for (int j = 0; j < expt_groups[tasks[i].type].size() && !flag_suc; ++j)
            {
                int expt_idx = expt_groups[tasks[i].type][j];
                double rand_num = (double)rand() / RAND_MAX;
                if (experts[expt_idx].num_idle_channel > 0 && rand_num < EPSILON)
                {
                    flag_suc = true;
                    assign_task(tasks[i], experts[expt_idx], i, expt_idx, env_tm);
                    flags_vis[i] = true;
                    break;
                }
            }
            if (!flag_suc)
            {
                // can only assign to not suitable expert
                for (int j = 0; j < experts.size() && !flag_suc; ++j)
                {
                    double rand_num = (double)rand() / RAND_MAX;
                    if (experts[j].num_idle_channel > 0 && rand_num < EPSILON)
                    {
                        flag_suc = true;
                        assign_task(tasks[i], experts[j], i, j, env_tm);
                        flags_vis[i] = true;
                    }
                }
            }
            // if still not assigned, just wait
        }
        // migrate or swap tasks, the operations only been applied to tasks on not suitable experts
        // and must keep sure task finally exected on suitable expert
        // firstly check if suitable experts available, then check if swap can be taken to make
        // some tasks been executed on suitable experts
        for (int i = 0; i < tasks.size(); ++i)
        {
            if (tasks[i].curr_migrate_count == 0 || flags_vis[i])
                continue;
            if (experts[tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1]].process_type_duras[tasks[i].type] ==
                monte_utils::EXPERT_NOT_GOOD_TIME)
            {
                for (int j = 0; j < expt_groups[tasks[i].type].size(); ++j)
                {
                    int expt_idx = expt_groups[tasks[i].type][j];
                    double rand_num = (double)rand() / RAND_MAX;
                    if (experts[expt_idx].num_idle_channel > 0 && rand_num < EPSILON)
                    {
                        assign_task(tasks[i], experts[expt_idx], i, expt_idx, env_tm);
                        flags_vis[i] = true;
                        break;
                    }
                }
            }
        }
        // try swap tasks
        for (int i = 0; i < tasks.size() - 1; ++i)
        {
            if (tasks[i].curr_migrate_count == 0 || flags_vis[i] || tasks[i].curr_migrate_count == monte_utils::TASK_MAX_MIGRATION)
                continue;
            int expt_idx_i = tasks[i].each_stay_expert_id[tasks[i].curr_migrate_count - 1];
            if (experts[expt_idx_i].process_type_duras[tasks[i].type] < monte_utils::EXPERT_NOT_GOOD_TIME)
                continue;
            for (int j = i + 1; j < tasks.size(); ++j)
            {
                if (tasks[j].curr_migrate_count == 0 || flags_vis[j] || tasks[j].curr_migrate_count == monte_utils::TASK_MAX_MIGRATION)
                    continue;
                int expt_idx_j = tasks[j].each_stay_expert_id[tasks[j].curr_migrate_count - 1];
                if (experts[expt_idx_j].process_type_duras[tasks[j].type] < monte_utils::EXPERT_NOT_GOOD_TIME)
                    continue;
                double rand_num = (double)rand() / RAND_MAX;
                // try swap
                if (rand_num < EPSILON && swap_check(tasks[i], tasks[j], experts[expt_idx_i], experts[expt_idx_j]))
                {
                    // both swap to suitable expert
                    swap_tasks(tasks[i], tasks[j], experts[expt_idx_i], experts[expt_idx_j], i, j, expt_idx_i, expt_idx_j, env_tm);
                    flags_vis[i] = true;
                    flags_vis[j] = true;
                    break;
                }
            }
        }

        env_tm++;
        // update expert flag
        for (int i = 0; i < experts.size(); ++i)
        {
            if (experts[i].num_idle_channel < monte_utils::EXPERT_MAX_PARALLEL)
                experts[i].busy_sum++;
        }
        if (env_tm - 1 == snap_shot_beg)
        {
            snap_shot_beg += take_snapshot_gap;
            snapshots.emplace_back(SnapShot(env_tm, tasks, experts, flags_finish));
        }
    }

    std::vector<std::vector<int>> result = extract_result(tasks, experts);
    double score = monte_metrics::score(tasks, experts);
    for (int i = 0; i < snapshots.size(); ++i)
        snapshots[i].score = score;
    return std::make_tuple(result, score, snapshots);
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
    char prefix[100] = "\0";
    strncpy(prefix, utils::PRED_RESULT_PREFIX, sizeof(utils::PRED_RESULT_PREFIX));
    utils::save_result(strcat(prefix, result_tm_stamp), result);
}

int main(int argc, char const *argv[])
{
    double best_score = 0;
    std::vector<SnapShot> snap_shots;
    // generate initial snapshots from the very beginning
    std::vector<std::vector<int>> expt_groups;
    std::cout << "Generate initial solutions..." << std::endl;
    for (int iter = 1; iter <= 1; ++iter)
    {
        srand(time(nullptr));
        std::vector<monte_utils::Task> tasks = monte_utils::load_tasks();
        std::sort(tasks.begin(), tasks.end(), [](const monte_utils::Task &a, const monte_utils::Task &b) -> bool {
            if (a.generate_tm != b.generate_tm)
                return a.generate_tm < b.generate_tm;
            else
                return a.task_id < b.task_id;
        });
        std::vector<monte_utils::Expert> experts = monte_utils::load_experts();
        expt_groups = group_experts(experts, monte_utils::NUM_TASK_TYPE);
        std::tuple<std::vector<std::vector<int>>, double, std::vector<SnapShot>> ret = run_alg(tasks, experts, std::vector<bool>(), expt_groups, 0, false);
        if (std::get<1>(ret) > best_score)
        {
            best_score = std::get<1>(ret);
            save_result(std::get<0>(ret));
        }
        // add snapshots
        std::cout << ">> Add snapshots " << std::get<2>(ret).size() << std::endl;
        snap_shots.insert(snap_shots.end(), std::get<2>(ret).begin(), std::get<2>(ret).end());
    }

    std::cout << "Generate initial solutions finish, snap_shots size=" << snap_shots.size() << std::endl;
    const int SNAP_SHOT_MAX_KEEP = 8;
    // iterations
    // after each iteration, new snapshots will be generated, and low score snapshots will be dropped
    std::cout << "Start iterations ..." << std::endl;
    for (int iter = 1; iter <= 1000; ++iter)
    {
        std::cout << ">>> Iter #" << iter << " best score=" << best_score << std::endl;
        std::sort(snap_shots.begin(), snap_shots.end(), [](const SnapShot &a, const SnapShot &b) -> bool {
            if(abs(a.score-b.score) > 0.01)
                return a.score < b.score;
            else if(a.snap_shot_tm != b.snap_shot_tm)
                return a.snap_shot_tm > b.snap_shot_tm;
            else
                return a.score < b.score;
        });
        if (snap_shots.size() > SNAP_SHOT_MAX_KEEP)
        {
            int remove_size = (int)snap_shots.size() - SNAP_SHOT_MAX_KEEP;
            snap_shots.erase(snap_shots.begin(), snap_shots.begin() + remove_size);
        }
        // try from each snapshots
        int shot_size = snap_shots.size();
        std::vector<std::vector<SnapShot>> tmp_snps(shot_size);
        std::vector<double> snap_scores(shot_size, 0);
        std::vector<std::vector<std::vector<int>>> snap_solutions(shot_size);
        for (int i = 0; i < shot_size; ++i)
        {
            std::tuple<std::vector<std::vector<int>>, double, std::vector<SnapShot>> ret = run_alg(snap_shots[i].tasks, snap_shots[i].experts,
                                                                                                   snap_shots[i].flags_finish,
                                                                                                   expt_groups, snap_shots[i].snap_shot_tm, true);
            snap_scores[i] = std::get<1>(ret);
            snap_solutions[i] = std::get<0>(ret);
            tmp_snps[i] = std::get<2>(ret);
        }
        // check best score and add snpshots
        int best_idx = 0;
        for (int i = 0; i < shot_size; ++i)
        {
            if (snap_scores[i] > best_score)
            {
                best_idx = i;
                best_score = snap_scores[i];
            }
        }
        std::cout << ">>>> Iter " << iter << " best score=" << best_score << std::endl;
        save_result(snap_solutions[best_idx]);
        for (int i = 0; i < shot_size; ++i)
            snap_shots.insert(snap_shots.end(), tmp_snps[i].begin(), tmp_snps[i].end());
        if (iter % 100 == 0)
            EPSILON *= 0.8;
    }
    return 0;
}
