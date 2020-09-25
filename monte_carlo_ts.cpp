/**
 * Monte Carlo Tree Search Algorithm
 * 
 */
#include "metrics.hpp"
#include <unistd.h>
#include <random>
#include <omp.h>

static std::vector<std::vector<int>> best_result; // each line contains three value: task id, expert id, start time
static double best_score = 0;
static std::default_random_engine random_gen;
static int force_migrate_max_exec_tm = 1000; // if task has executed on a expert for more than the value, the task must be forced to migrate
static const int MAX_NUM_CHILDREN = 5;

// Monte Carlo Tree Node Structure
// The Node records the state of tasks and experts at certain time stamp
struct MCTreeNode
{
    int env_tm;
    int num_sim;
    double reward_sum;
    MCTreeNode *parent;
    int num_finish_tasks;
    std::vector<utils::Task> task_status;
    std::vector<utils::Expert> expert_status;
    MCTreeNode *children_node[MAX_NUM_CHILDREN];
    int child_node_count;

    MCTreeNode() : env_tm(0), num_sim(0), reward_sum(0), num_finish_tasks(0), parent(nullptr), child_node_count(0)
    {
        std::fill(children_node, children_node + MAX_NUM_CHILDREN, nullptr);
    }

    ~MCTreeNode()
    {
        parent = nullptr;
        task_status.clear();
        expert_status.clear();
        for (int i = 0; i < child_node_count; ++i)
            children_node[i] = nullptr;
    }

    MCTreeNode &operator=(const MCTreeNode &node)
    {
        if (this != &node)
        {
            this->env_tm = node.env_tm;
            this->num_sim = 0;
            this->reward_sum = 0;
            this->parent = nullptr;
            this->num_finish_tasks = node.num_finish_tasks;
            this->task_status.resize(node.task_status.size());
            for (int i = 0; i < node.task_status.size(); ++i)
                this->task_status[i] = node.task_status[i];
            this->expert_status.resize(node.expert_status.size());
            for (int i = 0; i < node.expert_status.size(); ++i)
                this->expert_status[i] = node.expert_status[i];
        }
        return *this;
    }

    void add_new_child(MCTreeNode *child)
    {
        this->children_node[this->child_node_count] = child;
        this->child_node_count++;
    }

    /**
     * only remove child node from this node's children_node array, not free it
     */
    void remove_last_child()
    {
        --this->child_node_count;
        this->children_node[this->child_node_count] = nullptr;
    }

    void clear_children_nodes()
    {
        for (int i = 0; i < this->child_node_count; ++i)
            this->children_node[i] = nullptr;
        this->child_node_count = 0;
    }
};

MCTreeNode *init_root(std::vector<utils::Task> &tasks, std::vector<utils::Expert> &experts)
{
    MCTreeNode *root = new MCTreeNode();
    root->task_status.resize(tasks.size());
    root->expert_status.resize(experts.size());
    for (int i = 0; i < tasks.size(); ++i)
        root->task_status[i] = tasks[i];
    for (int j = 0; j < experts.size(); ++j)
        root->expert_status[j] = experts[j];
    return root;
}

/**
 * Group experts by their type 
 */
std::vector<std::vector<int>> group_expert_by_type(std::vector<utils::Expert> &experts, int num_types)
{
    std::vector<std::vector<int>> expert_groups(num_types);
    for (int i = 0; i < experts.size(); ++i)
    {
        for (int j = 0; j < experts[i].process_dura.size(); ++j)
        {
            if (experts[i].process_dura[j] < utils::EXPERT_NOT_GOOD_TIME)
                expert_groups[j].push_back(i);
        }
    }
    return expert_groups;
}

bool assign_to_expert(MCTreeNode *&node, int selected_task_idx, int env_tm, int selected_expert_idx)
{
    utils::Task *tsk = &node->task_status[selected_task_idx];
    if (tsk->task_curr_via_count == 0)
        tsk->start_process_tmpt = env_tm;
    int type = tsk->type;
    tsk->task_stay_due_tm[tsk->task_curr_via_count] = env_tm + node->expert_status[selected_expert_idx].process_dura[type] - 1;
    tsk->task_via_expert_idxs[tsk->task_curr_via_count] = selected_expert_idx;
    tsk->task_assign_expert_tm[tsk->task_curr_via_count] = env_tm;
    tsk->task_curr_via_count++;
    node->expert_status[selected_expert_idx].monte_assign_task(selected_task_idx);
    tsk = nullptr;
    return true;
}

bool assign_rand_expert(MCTreeNode *&node, int selected_task_idx, int env_tm, std::uniform_int_distribution<int> &dist, int max_num_assign_retry = 100)
{
    int random_action = dist(random_gen);
    // std::cout << "Task " << selected_task_idx << " first time assigned to expert is " << random_action << " need check...." << std::endl;
    node->env_tm = env_tm;
    // assign to expert to execute
    int assign_retry_count = 0;
    bool flag_assign = false;
    while (assign_retry_count++ < max_num_assign_retry)
    {
        flag_assign = node->expert_status[random_action].monte_assign_task(selected_task_idx);
        if (flag_assign)
            break;
    }
    if (!flag_assign)
        return false;
    // std::cout << "Task " << selected_task_idx << " after check, select expert = " << random_action << std::endl;
    utils::Task *tsk = &node->task_status[selected_task_idx];
    if (tsk->task_curr_via_count == 0)
        tsk->start_process_tmpt = env_tm;
    int task_type = tsk->type;
    tsk->task_stay_due_tm[tsk->task_curr_via_count] = env_tm + node->expert_status[random_action].process_dura[task_type] - 1;
    tsk->task_via_expert_idxs[tsk->task_curr_via_count] = random_action;
    tsk->task_assign_expert_tm[tsk->task_curr_via_count] = env_tm;
    tsk->task_curr_via_count++;
    tsk = nullptr;
    return true;
}

bool assign_suit_expert(MCTreeNode *&node, int selected_task_idx, int env_tm,
                        std::vector<std::vector<int>> &expt_group, int max_num_assign_retry = 100)
{
    // Choose from the suitable expert group
    node->env_tm = env_tm;
    utils::Task *selected_tsk = &node->task_status[selected_task_idx];
    int task_type = selected_tsk->type;
    std::uniform_int_distribution<int> dist_suit_group(0, expt_group[task_type].size() - 1);
    int random_choose_expt = expt_group[task_type][dist_suit_group(random_gen)];
    int num_retry = 0;
    bool is_found_suit_expt = true;
    while (num_retry++ < max_num_assign_retry)
    {
        is_found_suit_expt = node->expert_status[random_choose_expt].monte_assign_task(selected_task_idx);
        if (is_found_suit_expt)
            break;
    }
    if (is_found_suit_expt)
    {
        // std::cout << "Task " << selected_task_idx << " choose suit expert " << random_choose_expt << std::endl;
        // assign the task to the expert
        if (selected_tsk->task_curr_via_count == 0)
            selected_tsk->start_process_tmpt = env_tm;
        selected_tsk->task_stay_due_tm[selected_tsk->task_curr_via_count] = env_tm + node->expert_status[random_choose_expt].process_dura[task_type] - 1;
        selected_tsk->task_via_expert_idxs[selected_tsk->task_curr_via_count] = random_choose_expt;
        selected_tsk->task_assign_expert_tm[selected_tsk->task_curr_via_count] = env_tm;
        selected_tsk->task_curr_via_count++;
    }
    selected_tsk = nullptr;
    return is_found_suit_expt;
}

/**
 * Exploit from current monte carlo tree node, add child nodes
 * @param expt_group: the expert index grouped by suitable types
 * @param num_expand: the max expand child node count
 * @param max_num_assign_retry: the random selected expert may not available, if too many try failed, the expand operation is failed
 * @param possible_beg_not_wait: when reaching the generating time of a task, the possibility range of not waitting
 * @param possible_percent_stick_curr: the possibility for choosing stick to current expert, continuing executing
 * @param possible_favor_suit_expert: the possibility of assigning current task to suitable expert to process
 */
bool expand(MCTreeNode *&root, std::vector<std::vector<int>> &expt_group, int num_max_retry = 10, int num_expand = MAX_NUM_CHILDREN - 1,
            int possible_beg_not_wait = 95, int possible_percent_stick_curr = 98, int possible_favor_suit_expert = 98)
{
    // randomly select valid action to expand new nodes, simulated to terminal state
    // and backpropgate the rewards
    int env_tm = root->env_tm + 1;
    std::uniform_int_distribution<int> dist_percent(1, 100);
    std::uniform_int_distribution<int> dist(0, root->task_status.size() - 1);
    std::uniform_int_distribution<int> dist_action_no_wait(0, root->expert_status.size() - 1);
    // std::cout << "Start expand, env_tm = " << env_tm << std::endl;
    while (num_expand-- > 0)
    {
        int selected_task_idx = dist(random_gen);
        // After randomly select a task, firstly, the time should be checked
        // Only when the env_tm is ge task's genereate time, the task can task action
        // After checking time, the task can choose wait if it has not been assigned,
        // or assign/reassign to an expert  or go on executing on current expert
        if (root->task_status[selected_task_idx].tm_stamp > env_tm)
        {
            // std::cout << "\t selected task " << selected_task_idx << " tm stamp=" << root->task_status[selected_task_idx].tm_stamp
            //           << " not generated yet at time " << env_tm << std::endl;
            continue;
        }
        MCTreeNode *child = new MCTreeNode();
        *child = *root;
        child->parent = root;
        child->env_tm = env_tm;
        root->add_new_child(child);
        // random generate integer in [0, number experts], where the last number used as wait,
        // the wait action can only be taken when the task has not been assigned to any expert
        if (root->task_status[selected_task_idx].task_via_expert_idxs[0] == -1)
        {
            // std::cout << "Task " << selected_task_idx << " first time assigned to expert or wait" << std::endl;
            // the task has not been assigned to any expert, can choose to wait
            int possible_assign_wait = dist_percent(random_gen);
            // choosing assigning to experts
            if (possible_assign_wait <= possible_beg_not_wait)
            {
                // The possibility of choosing expert to executing should favor the suitable expert
                int rand_choose_suit_expt = dist_percent(random_gen);
                if (rand_choose_suit_expt <= possible_favor_suit_expert)
                {
                    // Choose from the suitable expert group
                    if (!assign_suit_expert(child, selected_task_idx, env_tm, expt_group))
                    {
                        // if the above process not found suit expert or the random action is choos other experts that not be
                        // suitable for processing current task
                        if (!assign_rand_expert(child, selected_task_idx, env_tm, dist_action_no_wait))
                        {
                            // there may be exists experts available that have not been chosen
                            // simply try each expert one by one
                            utils::Task *tsk = &child->task_status[selected_task_idx];
                            int task_type = tsk->type;
                            std::vector<int> avail_suit_expert_idxs, avail_no_suit_expert_idxs;
                            for (int k = 0; k < child->expert_status.size(); ++k)
                            {
                                if (child->expert_status[k].num_avail > 0 &&
                                    child->expert_status[k].process_dura[task_type] < utils::EXPERT_NOT_GOOD_TIME)
                                    avail_suit_expert_idxs.push_back(k);
                                else if (child->expert_status[k].num_avail > 0)
                                    avail_no_suit_expert_idxs.push_back(k);
                            }
                            if (!avail_suit_expert_idxs.empty())
                            {
                                std::sort(avail_suit_expert_idxs.begin(), avail_suit_expert_idxs.end(), [child, task_type](const int a, const int b) -> bool {
                                    if (child->expert_status[a].process_dura[task_type] != child->expert_status[b].process_dura[task_type])
                                        return child->expert_status[a].process_dura[task_type] < child->expert_status[b].process_dura[task_type];
                                    else if (child->expert_status[a].num_avail != child->expert_status[b].num_avail)
                                        return child->expert_status[a].num_avail > child->expert_status[b].num_avail;
                                    else
                                        return child->expert_status[a].id < child->expert_status[b].id;
                                });
                                assign_to_expert(child, selected_task_idx, env_tm, avail_suit_expert_idxs[0]);
                            }
                            else if (!avail_no_suit_expert_idxs.empty())
                            {
                                assign_to_expert(child, selected_task_idx, env_tm, avail_no_suit_expert_idxs[0]);
                            }
                            else
                            {
                                avail_suit_expert_idxs.clear();
                                avail_no_suit_expert_idxs.clear();
                                std::cout << __LINE__
                                          << " simulation terminate, reason=task assigned to expert failed at init stage, failed tasks = "
                                          << selected_task_idx << std::endl;
                                root->remove_last_child();
                                delete child;
                                child = nullptr;
                                return false;
                            }
                            avail_suit_expert_idxs.clear();
                            avail_no_suit_expert_idxs.clear();
                            tsk = nullptr;
                        }
                    }
                }
            }
        }
        else if (env_tm -
                     child->task_status[selected_task_idx].task_assign_expert_tm[child->task_status[selected_task_idx].task_curr_via_count - 1] >=
                 force_migrate_max_exec_tm)
        {
            // when a task executing on a expert for too long time, the task must be forced to migrate
            // if the task has reached the max migration restrict, the simulation is failed
            utils::Task *tsk = &child->task_status[selected_task_idx];
            if (tsk->task_curr_via_count == utils::TASK_MAX_MIGRATION)
            {
                std::cout << __LINE__ << " simulation terminate, reason=task at a non-suitable expert with max migration" << std::endl;
                root->remove_last_child();
                delete child;
                child = nullptr;
                tsk = nullptr;
                return false;
            }
            else
            {
                // migrate to other experts, if next migration reach the max restriction, must choose suitable expert,
                // if can not found, return false. else the choice just need to favor the suitable expert
                if (tsk->task_curr_via_count == utils::TASK_MAX_MIGRATION - 1)
                {
                    // force choosing suitable expert
                    std::vector<int> avail_expert_idx;
                    for (int i = 0; i < child->expert_status.size(); ++i)
                    {
                        if (child->expert_status[i].num_avail > 0)
                            avail_expert_idx.push_back(i);
                    }
                    if (!avail_expert_idx.empty())
                    {
                        std::uniform_int_distribution<int> dist_rand_avail(0, avail_expert_idx.size() - 1);
                        int choice_idx = dist_rand_avail(random_gen);
                        int prev_expert_idx = tsk->task_via_expert_idxs[tsk->task_curr_via_count - 1];
                        assign_to_expert(child, selected_task_idx, env_tm, choice_idx); // the choosen expert is available
                        // release task from current expert
                        child->expert_status[prev_expert_idx].monte_release_task(selected_task_idx);
                        tsk->each_stay_dura.push_back(env_tm - tsk->task_assign_expert_tm[tsk->task_curr_via_count - 2]);
                    }
                    else
                    {
                        // no available suit expert
                        std::cout << __LINE__ << " simulation terminate, reason=next stage is max migration restrict, not found suitable one" << std::endl;
                        root->remove_last_child();
                        delete child;
                        child = nullptr;
                        tsk = nullptr;
                        return false;
                    }
                }
                else
                {
                    int rand_choose_suit_percent = dist_percent(random_gen);
                    if (rand_choose_suit_percent <= possible_favor_suit_expert)
                    {
                        // if try assigning suit expert failed, then will try assigning rand expert
                        int prev_expert_idx = tsk->task_via_expert_idxs[tsk->task_curr_via_count - 1];
                        if (!assign_suit_expert(child, selected_task_idx, env_tm, expt_group))
                        {
                            if (!assign_rand_expert(child, selected_task_idx, env_tm, dist_action_no_wait))
                            {
                                std::cout << __LINE__ << " simulation terminate, reason=task force migrate, not found suitable next expert" << std::endl;
                                root->remove_last_child();
                                delete child;
                                child = nullptr;
                                tsk = nullptr;
                                return false;
                            }
                        }
                        // release task from current expert
                        child->expert_status[prev_expert_idx].monte_release_task(selected_task_idx);
                        tsk->each_stay_dura.push_back(env_tm - tsk->task_assign_expert_tm[tsk->task_curr_via_count - 2]);
                    }
                }
            }
            tsk = nullptr;
        }
        else
        {
            // The task has been assigned to at least one expert, so, the possible actions are continuing executing and migration
            // The possibility should favor current assigned expert
            // At here 95% for current expert and 5% for migrating to other expert, if the task has reached the max migration restricted
            // then the task can only choose continuing executing on current expert
            int rand_curr_migrate = dist_percent(random_gen);
            utils::Task *selected_task = &child->task_status[selected_task_idx];
            int current_assigned_expt_idx = selected_task->task_via_expert_idxs[selected_task->task_curr_via_count - 1];
            if (rand_curr_migrate <= possible_percent_stick_curr || selected_task->task_curr_via_count == utils::TASK_MAX_MIGRATION)
            {
                // choose to continue executing on current expert
                // check if the tasks finished at current time slot
                // std::cout << "Task " << selected_task_idx << " continuing executing on expert " << std::endl;
                if (env_tm - selected_task->task_assign_expert_tm[selected_task->task_curr_via_count - 1] >= force_migrate_max_exec_tm)
                {
                    std::cout << __LINE__
                              << " simulation terminate, reason=continuing executing on current expert, reach max exec time restrict" << std::endl;
                    root->remove_last_child();
                    delete child;
                    child = nullptr;
                    selected_task = nullptr;
                    return false;
                }
                else if (env_tm == selected_task->task_stay_due_tm[selected_task->task_curr_via_count - 1])
                {
                    // Task finished on the expert
                    selected_task->finish_tmpt = env_tm;
                    child->expert_status[current_assigned_expt_idx].monte_release_task(selected_task_idx);
                    selected_task->each_stay_dura.push_back(env_tm - selected_task->task_assign_expert_tm[selected_task->task_curr_via_count - 1]);
                    // std::cout << "Task " << selected_task_idx << " finished" << std::endl;
                }
            }
            else
            {
                // choose other experts to migrate
                int random_action = dist_action_no_wait(random_gen);
                // std::cout << "Task " << selected_task_idx << " migrate" << std::endl;
                // check if the expert is available, if not, random choosing another
                int retry_count = 0;
                bool flag_assign = false;
                while (retry_count < num_max_retry)
                {
                    while (random_action == current_assigned_expt_idx)
                        random_action = dist_action_no_wait(random_gen);
                    retry_count++;
                    flag_assign = child->expert_status[random_action].monte_assign_task(selected_task_idx);
                    if (flag_assign)
                        break;
                }
                if (!flag_assign)
                {
                    // std::cout << __LINE__ << " simulation terminate, reason=normal assigned failed" << std::endl;
                    // return false;
                    // if migration failed, continuing executing on current expert, need to check if the task finished
                    if (env_tm - selected_task->task_assign_expert_tm[selected_task->task_curr_via_count - 1] >= force_migrate_max_exec_tm)
                    {
                        std::cout << __LINE__ << " simulation terminate, reason=continuing executing on current expert, reach max exec time restrict"
                                  << std::endl;
                        root->remove_last_child();
                        delete child;
                        child = nullptr;
                        selected_task = nullptr;
                        return false;
                    }
                    else if (env_tm == selected_task->task_stay_due_tm[selected_task->task_curr_via_count - 1])
                    {
                        // task finished at current time
                        selected_task->finish_tmpt = env_tm;
                        child->expert_status[current_assigned_expt_idx].monte_release_task(selected_task_idx);
                        selected_task->each_stay_dura.push_back(env_tm - selected_task->task_assign_expert_tm[selected_task->task_curr_via_count - 1]);
                    }
                }
                else
                {
                    // std::cout << "Task " << selected_task_idx << " migrate to expert " << random_action << std::endl;
                    // Must check the action is valid, the next expert to migrate may not have space
                    int prev_expert_idx = selected_task->task_via_expert_idxs[selected_task->task_curr_via_count - 1];
                    // record the duration of staying in previous expert
                    selected_task->each_stay_dura.push_back(env_tm - selected_task->task_assign_expert_tm[selected_task->task_curr_via_count - 1]);
                    // record new information of new migrated expert
                    selected_task->task_via_expert_idxs[selected_task->task_curr_via_count] = random_action;
                    int task_type = child->task_status[selected_task_idx].type;
                    selected_task->task_stay_due_tm[selected_task->task_curr_via_count] = env_tm + child->expert_status[random_action].process_dura[task_type] - 1;
                    selected_task->task_assign_expert_tm[selected_task->task_curr_via_count] = env_tm;
                    selected_task->task_curr_via_count++;
                    child->expert_status[prev_expert_idx].monte_release_task(selected_task_idx);
                }
            }
            selected_task = nullptr;
        }
        child = nullptr;
    }
    // When no random selected tasks is generated at current time, no action
    if (root->child_node_count == 0)
    {
        MCTreeNode *child = new MCTreeNode();
        *child = *root;
        child->env_tm = env_tm;
        child->parent = root;
        root->add_new_child(child);
        child = nullptr;
    }
    // For each child nodes, update the record variables of experts, e.t. busy time  for metrics calculating
    for (int i = 0; i < root->child_node_count; ++i)
    {
        for (int j = 0; j < root->children_node[i]->expert_status.size(); ++j)
            root->children_node[i]->expert_status[j].monte_one_tick();
    }
    return true;
}

/**
 * Simulate from current state until the terminal state
 * While reaching the terminal state, the reward will be calculated and backpropagate upward
 * This function do simulation once, the child nodes created during simulation will be released
 */
void simulate(MCTreeNode *root, std::vector<std::vector<int>> &expt_groups)
{
    MCTreeNode *curr_node = root;
    bool reach_end = false, expand_flag = true;
    std::cout << "Start simulation..." << std::endl;
    int simu_depth = 1;
    sleep(1);
    while (!reach_end && expand_flag)
    {
        simu_depth++;
        expand_flag = expand(curr_node, expt_groups, 1);
        if (!expand_flag)
            break;
        MCTreeNode *tmp = curr_node;
        curr_node = curr_node->children_node[0];
        if (tmp != root)
            delete tmp;
        tmp = nullptr;
        if (curr_node->num_finish_tasks == curr_node->task_status.size())
            reach_end = true;
    }
    // the simulation may be failed during the expand procedure
    if (!expand_flag)
    {
        root->num_sim++;
        if (curr_node != root)
        {
            curr_node->clear_children_nodes();
            delete curr_node;
        }
        root->clear_children_nodes();
        std::cout << "Monte Carlo simulation terminated at non finish stat, simulation terminated at depth " << simu_depth << std::endl;
        return;
    }
    std::cout << "Monte Carlo simulation reach end" << std::endl;
    // Calculating score and backpropagate
    std::vector<double> expt_workloads(curr_node->expert_status.size(), 0),
        task_resp_tmout(curr_node->task_status.size(), 0), exec_eff(curr_node->task_status.size(), 0);
    for (int i = 0; i < curr_node->task_status.size(); ++i)
    {
        task_resp_tmout[i] = metrics::task_response_timeout(curr_node->task_status[i]);
        exec_eff[i] = metrics::task_exec_efficiency(curr_node->task_status[i]);
    }
    for (int i = 0; i < curr_node->expert_status.size(); ++i)
        expt_workloads[i] = metrics::expert_workload(curr_node->expert_status[i]);
    double score = metrics::score(expt_workloads, task_resp_tmout, exec_eff);
    if (score > best_score)
    {
        // The result is better than current found best score, record it
        std::cout << "Monte Carlo simulation, found better solution, score=" << score << std::endl;
        best_score = score;
        best_result.clear();
        for (utils::Task &tsk : curr_node->task_status)
        {
            for (int i = 0; i < utils::EXPERT_MAX_PARALLEL; ++i)
            {
                if (tsk.task_assign_expert_tm[i] != -1)
                    best_result.emplace_back(std::vector<int>({tsk.task_id, tsk.task_via_expert_idxs[i], tsk.task_assign_expert_tm[i]}));
                else
                    break;
            }
        }
        std::sort(best_result.begin(), best_result.end(), [](std::vector<int> &a, std::vector<int> &b) -> bool {
            if (a[2] != b[2])
                return a[2] < b[2];
            else
                return a[0] < b[0];
        });
    }
    // Backpropagate
    if (curr_node != root)
    {
        curr_node->clear_children_nodes();
        delete curr_node;
    }
    root->num_sim++;
    root->reward_sum += score;
    root->clear_children_nodes();
}

/**
 * Monte Carlo Tree Search algorithm method
 * The algorithm contains four basic operations:
 *  1. Expansion, at the beginning of each iteration, the algorithm need to select the best leaf node so far to expand new child nodes
 *              At the initial state only root state, the expand operations is executed on root node
 *  2. Simulation, after expand some child nodes, the algorithm will simulate many times from the child node till the terminal state
 *  3. Backpropagate, when a simulation process reached the terminal state, the score will be calculated, if score is better than
 *              the best score so far, the score and the whole transition will be recorded.
 *  4. Selection, at the very initial state, the only choice is the root node, and after the above procedures, the best leaf node will be
 *              selected for next iteration
 */
void run_alg(MCTreeNode *root, std::vector<std::vector<int>> &expert_groups, int max_iter = 1000,
             int min_num_expand_child = MAX_NUM_CHILDREN, int num_simulate_each = 100)
{
    std::vector<MCTreeNode *> leaf_nodes;
    std::default_random_engine random_gen;
    leaf_nodes.push_back(root);
    std::cout << "Monte Carlo alg started...." << std::endl;
    int num_epoch = 1;
    while (max_iter-- > 0)
    {
        sleep(1);
        std::cout << "Iteration " << num_epoch++ << ", leaf node count= " << leaf_nodes.size() << std::endl;
        MCTreeNode *best_leaf_node = leaf_nodes[0];
        for (MCTreeNode *node : leaf_nodes)
        {
            if ((node->reward_sum / (node->num_sim + __DBL_EPSILON__)) >
                (best_leaf_node->reward_sum / best_leaf_node->num_sim + __DBL_EPSILON__))
            {
                best_leaf_node = node;
            }
        }
        // expand best leaf node and simulate from children nodes of the best leaf node, backpropagate and update
        std::cout << "In main iteration, expanding best leaf node...." << std::endl;
        for (int i = 0; i < min_num_expand_child - 1; ++i)
            expand(best_leaf_node, expert_groups);
        std::cout << "Expanding best leaf node finish." << std::endl;
        sleep(1);
        // remove from leaf_nodes record, and add new leaf nodes
        for (int i = 0; i < leaf_nodes.size(); ++i)
        {
            if (leaf_nodes[i] == best_leaf_node)
            {
                leaf_nodes.erase(leaf_nodes.begin() + i);
                break;
            }
        }
        for (int i = 0; i < best_leaf_node->child_node_count; ++i)
            leaf_nodes.push_back(best_leaf_node->children_node[i]);

        if (best_leaf_node->child_node_count > 0)
        {
            // simulate from the child nodes till terminate state, calc score and backpropagate
            std::cout << "\t start simulations from child nodes, total num child nodes=" << best_leaf_node->child_node_count << " ...." << std::endl;
            sleep(1);
            std::uniform_int_distribution<int> rand_dist(0, best_leaf_node->child_node_count - 1);
            for (int i = 0; i < num_simulate_each; ++i)
            {
                for (int j = 0; j < best_leaf_node->child_node_count; ++j)
                {
                    simulate(best_leaf_node->children_node[j], expert_groups);
                }
            }
        }
    }
}

int main(int argc, char const *argv[])
{
    std::vector<utils::Task> tasks = utils::load_work_order();
    std::vector<utils::Expert> experts = utils::load_expert_process_duras();
    int num_types = experts[0].num_types();
    std::vector<std::vector<int>> expert_type_group = group_expert_by_type(experts, num_types);
    MCTreeNode *root = init_root(tasks, experts);
    run_alg(root, expert_type_group);
    return 0;
}
