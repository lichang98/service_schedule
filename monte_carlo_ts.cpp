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
    std::vector<MCTreeNode *> children_nodes;

    MCTreeNode() : env_tm(0), num_sim(0), reward_sum(0), num_finish_tasks(0), parent(nullptr) {}

    MCTreeNode &operator=(const MCTreeNode &node)
    {
        this->env_tm = node.env_tm;
        this->num_sim = 0;
        this->reward_sum = 0;
        this->parent = nullptr;
        this->num_finish_tasks = node.num_finish_tasks;
        this->task_status.resize(node.task_status.size());
        for (int i = 0; i < task_status.size(); ++i)
            this->task_status[i] = node.task_status[i];
        this->expert_status.resize(node.expert_status.size());
        for (int i = 0; i < expert_status.size(); ++i)
            this->expert_status[i] = node.expert_status[i];
        this->children_nodes.resize(node.children_nodes.size());
        for (int i = 0; i < children_nodes.size(); ++i)
            this->children_nodes[i] = node.children_nodes[i];
        return *this;
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
 * Exploit from current monte carlo tree node, add child nodes
 * @param num_expand: the max expand child node count
 * @param max_num_assign_retry: the random selected expert may not available, if too many try failed, the expand operation is failed
 * @param possible_beg_not_wait: when reaching the generating time of a task, the possibility range of not waitting
 * @param possible_percent_stick_curr: the possibility for choosing stick to current expert, continuing executing
 */
bool expand(MCTreeNode *root, int num_expand = 10, int max_num_assign_retry = 100, int possible_beg_not_wait = 90, int possible_percent_stick_curr = 100)
{
    // randomly select valid action to expand new nodes, simulated to terminal state
    // and backpropgate the rewards
    int env_tm = root->env_tm + 1;
    std::uniform_int_distribution<int> dist_percent(1, 100);
    std::uniform_int_distribution<int> dist(0, root->task_status.size() - 1);
    std::uniform_int_distribution<int> dist_action_no_wait(0, root->expert_status.size() - 1);
    std::cout << "Start expand, env_tm = " << env_tm << std::endl;
    while (num_expand-- > 0)
    {
        int selected_task_idx = dist(random_gen);
        // After randomly select a task, firstly, the time should be checked
        // Only when the env_tm is ge task's genereate time, the task can task action
        // After checking time, the task can choose wait if it has not been assigned,
        // or assign/reassign to an expert  or go on executing on current expert
        if (root->task_status[selected_task_idx].tm_stamp < env_tm)
            continue;
        MCTreeNode *child = new MCTreeNode();
        *child = *root;
        child->parent = root;
        child->env_tm = env_tm;
        child->children_nodes.clear();
        root->children_nodes.push_back(child);
        // random generate integer in [0, number experts], where the last number used as wait,
        // the wait action can only be taken when the task has not been assigned to any expert
        if (root->task_status[selected_task_idx].task_via_expert_idxs[0] == -1)
        {
            std::cout << "Task " << selected_task_idx << " first time assigned to expert or wait" << std::endl;
            // the task has not been assigned to any expert, can choose to wait
            int possible_assign_wait = dist_percent(random_gen);
            // choosing assigning to experts
            if (possible_assign_wait <= possible_beg_not_wait)
            {
                int random_action = dist_action_no_wait(random_gen);
                std::cout << "Task " << selected_task_idx << " first time assigned to expert is " << random_action << " need check...." << std::endl;
                child->env_tm = env_tm;
                // assign to expert to execute
                while (max_num_assign_retry-- > 0 && !child->expert_status[random_action].monte_assign_task(selected_task_idx))
                    random_action = dist_action_no_wait(random_gen);
                if (max_num_assign_retry == 0)
                    return false;
                std::cout << "Task " << selected_task_idx << " after check, select expert = " << random_action << std::endl;
                child->task_status[selected_task_idx].start_process_tmpt = env_tm;
                int task_type = child->task_status[selected_task_idx].type;
                child->task_status[selected_task_idx].task_stay_due_tm[0] = env_tm + child->expert_status[random_action].process_dura[task_type] - 1;
                child->task_status[selected_task_idx].task_via_expert_idxs[0] = random_action;
                child->task_status[selected_task_idx].task_assign_expert_tm[0] = env_tm;
                child->task_status[selected_task_idx].task_curr_via_count++;
            }
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
                std::cout << "Task " << selected_task_idx << " continuing executing on expert " << std::endl;
                if (env_tm == selected_task->task_stay_due_tm[selected_task->task_curr_via_count - 1])
                {
                    // Task finished on the expert
                    selected_task->finish_tmpt = env_tm;
                    child->expert_status[current_assigned_expt_idx].monte_release_task(selected_task_idx);
                    selected_task->each_stay_dura.push_back(env_tm - selected_task->task_assign_expert_tm[selected_task->task_curr_via_count - 1]);
                    std::cout << "Task " << selected_task_idx << " finished" << std::endl;
                }
            }
            else
            {
                // choose other experts to migrate
                int random_action = dist_action_no_wait(random_gen);
                std::cout << "Task " << selected_task_idx << " migrate" << std::endl;
                // check if the expert is available, if not, random choosing another
                while (random_action == current_assigned_expt_idx || !child->expert_status[random_action].monte_assign_task(selected_task_idx))
                    random_action = dist_action_no_wait(random_gen);
                std::cout << "Task " << selected_task_idx << " migrate to expert " << random_action << std::endl;
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
    }

    // When no random selected tasks is generated at current time, no action
    if (root->children_nodes.empty())
    {
        MCTreeNode *child = new MCTreeNode();
        *child = *root;
        child->env_tm = env_tm;
        child->children_nodes.clear();
        child->parent = root;
        root->children_nodes.push_back(child);
    }
    // For each child nodes, update the record variables of experts, e.t. busy time  for metrics calculating
    for (MCTreeNode *child : root->children_nodes)
    {
        for (int i = 0; i < child->expert_status.size(); ++i)
            child->expert_status[i].monte_one_tick();
    }
    return true;
}

/**
 * Simulate from current state until the terminal state
 * While reaching the terminal state, the reward will be calculated and backpropagate upward
 * This function do simulation once, the child nodes created during simulation will be released
 */
void simulate(MCTreeNode *root)
{
    MCTreeNode *curr_node = root;
    bool reach_end = false, expand_flag = true;
    std::cout << "Start simulation..." << std::endl;
    while (!reach_end && expand_flag)
    {
        expand_flag = expand(curr_node, 1);
        if (!expand_flag)
            break;
        MCTreeNode *tmp = curr_node;
        curr_node = curr_node->children_nodes[0];
        if (tmp != root)
            delete tmp;
        if (curr_node->num_finish_tasks == curr_node->task_status.size())
            reach_end = true;
    }
    // the simulation may be failed during the expand procedure
    if (!expand_flag)
    {
        root->num_sim++;
        if (curr_node != root)
            delete curr_node;
        root->children_nodes.clear();
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
        delete curr_node;
    root->num_sim++;
    root->reward_sum += score;
    root->children_nodes.clear();
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
void run_alg(MCTreeNode *root, int max_iter = 1000, int min_num_expand_child = 10, int num_simulate_each = 100)
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
        for (int i = 0; i < min_num_expand_child; ++i)
            expand(best_leaf_node);
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
        for (MCTreeNode *child_node : best_leaf_node->children_nodes)
            leaf_nodes.push_back(child_node);

        if (!best_leaf_node->children_nodes.empty())
        {
            // simulate from the child nodes till terminate state, calc score and backpropagate
            std::cout << "\t start simulations from child nodes...." << std::endl;
            sleep(1);
            std::uniform_int_distribution<int> rand_dist(0, best_leaf_node->children_nodes.size() - 1);
            for (int i = 0; i < num_simulate_each; ++i)
            {
                for (int j = 0; j < best_leaf_node->children_nodes.size(); ++j)
                {
                    simulate(best_leaf_node->children_nodes[j]);
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
    MCTreeNode *root = init_root(tasks, experts);
    run_alg(root);
    return 0;
}
