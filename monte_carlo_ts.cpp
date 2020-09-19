/**
 * Monte Carlo Tree Search Algorithm
 * 
 */
#include "metrics.hpp"
#include <random>

static std::vector<std::vector<int>> best_result; // each line contains three value: task id, expert id, start time
static double best_score = 0;

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
 */
void expand(MCTreeNode *root, int num_expand = 10)
{
    // randomly select valid action to expand new nodes, simulated to terminal state
    // and backpropgate the rewards
    int env_tm = root->env_tm + 1;
    std::default_random_engine random_gen;
    std::uniform_int_distribution<int> dist(0, root->task_status.size() - 1);
    std::uniform_int_distribution<int> dist_action_with_wait(0, root->expert_status.size());
    std::uniform_int_distribution<int> dist_action_no_wait(0, root->expert_status.size() - 1);

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
        child->children_nodes.clear();
        root->children_nodes.push_back(child);
        // random generate integer in [0, number experts], where the last number used as wait,
        // the wait action can only be taken when the task has not been assigned to any expert
        if (root->task_status[selected_task_idx].task_via_expert_idxs[0] == -1)
        {
            // the task has not been assigned to any expert, can choose to wait
            int random_action = dist_action_with_wait(random_gen);
            if (random_action == root->expert_status.size())
            {
                // wait
                child->env_tm = env_tm;
            }
            else
            {
                // assign to expert to execute
                child->expert_status[random_action].monte_assign_task(selected_task_idx);
                child->task_status[selected_task_idx].start_process_tmpt = env_tm;
                int task_type = child->task_status[selected_task_idx].type;
                child->task_status[selected_task_idx].task_stay_due_tm[0] = env_tm + child->expert_status[random_action].process_dura[task_type];
                child->task_status[selected_task_idx].task_via_expert_idxs[0] = random_action;
                child->task_status[selected_task_idx].task_curr_via_count++;
            }
        }
        else
        {
            // The task has been assigned to at least one expert, so, the possible actions are continuing executing and migration
            int random_action = dist_action_no_wait(random_gen);
            utils::Task *selected_task = &child->task_status[selected_task_idx];
            if (random_action == selected_task->task_via_expert_idxs[selected_task->task_curr_via_count - 1])
            {
                // Continuing executing on this expert, check if the task finish
                if (env_tm == selected_task->task_stay_due_tm[selected_task->task_curr_via_count - 1])
                {
                    // Task finished on the expert
                    selected_task->finish_tmpt = env_tm;
                    child->expert_status[random_action].monte_release_task(selected_task_idx);
                }
            }
            else
            {
                // Migrate to new expert
                if (child->expert_status[random_action].monte_assign_task(selected_task_idx))
                {
                    // Must check the action is valid, the next expert to migrate may not have space
                    int prev_expert_idx = selected_task->task_via_expert_idxs[selected_task->task_curr_via_count - 1];
                    child->expert_status[prev_expert_idx].monte_release_task(selected_task_idx);
                }
            }
        }
    }

    // For each child nodes, update the record variables of experts, e.t. busy time  for metrics calculating
    for (MCTreeNode *child : root->children_nodes)
    {
        for (int i = 0; i < child->expert_status.size(); ++i)
            child->expert_status[i].monte_one_tick();
    }
    if (root->children_nodes.empty())
    {
        // If may occures at the beginning when no task reach the generating time or when all tasks finish
        if (root->num_finish_tasks == root->task_status.size())
        {
            // all tasks finish
        }
        else
        {
            // no task reach generating time
            MCTreeNode *child = new MCTreeNode();
            *child = *root;
            child->env_tm++;
            root->children_nodes.push_back(child);
            child->parent = root;
            child->children_nodes.clear();
        }
    }
}

/**
 * Simulate from current state until the terminal state
 * While reaching the terminal state, the reward will be calculated and backpropagate upward
 * This function do simulation once, the child nodes created during simulation will be released
 */
void simulate(MCTreeNode *root)
{
    MCTreeNode *curr_node = root;
    bool reach_end = false;
    while (!reach_end)
    {
        // Use Expand operation, only one child node will be expanded
        expand(curr_node, 1);
        curr_node = curr_node->children_nodes[0];
        if (curr_node->num_finish_tasks == curr_node->task_status.size())
        {
            reach_end = true;
        }
    }
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
        best_score = score;
        best_result.clear();
        for (utils::Task &tsk : curr_node->task_status)
        {
            for (int i = 0; i < utils::EXPERT_MAX_PARALLEL; ++i)
            {
                if (tsk.task_assign_expert_tm[i] != -1)
                    best_result.emplace_back(std::vector<int>({tsk.task_id, tsk.task_via_expert_idxs[i], tsk.task_assign_expert_tm[i]}));
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
    MCTreeNode *child = curr_node;
    curr_node = curr_node->parent;
    while (curr_node != root)
    {
        delete child;
        child = curr_node;
        curr_node = curr_node->parent;
    }
    curr_node->num_sim++;
    curr_node->reward_sum += score;
    curr_node->children_nodes.clear();
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
    while (max_iter-- > 0)
    {
        MCTreeNode *best_leaf_node = leaf_nodes[0];
        for (MCTreeNode *node : leaf_nodes)
        {
            if ((node->reward_sum / (node->num_sim + __DBL_EPSILON__)) <
                (best_leaf_node->reward_sum / best_leaf_node->num_sim + __DBL_EPSILON__))
            {
                best_leaf_node = node;
            }
        }
        // expand best leaf node and simulate from children nodes of the best leaf node, backpropagate and update
        int num_child_expand = 0, max_retry = 10;
        while (max_retry-- > 0 && num_child_expand < min_num_expand_child)
        {
            expand(best_leaf_node);
            num_child_expand = best_leaf_node->children_nodes.size();
        }
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
            std::uniform_int_distribution<int> rand_dist(0, best_leaf_node->children_nodes.size() - 1);
            for (int i = 0; i < num_simulate_each; ++i)
            {
                int rand_select_node_idx = rand_dist(random_gen);
                simulate(best_leaf_node->children_nodes[rand_select_node_idx]);
            }
        }
    }
}

int main(int argc, char const *argv[])
{
    std::vector<utils::Task> tasks = utils::load_work_order();
    std::vector<utils::Expert> experts = utils::load_expert_process_duras();
    int num_types = experts[0].num_types();

    return 0;
}
