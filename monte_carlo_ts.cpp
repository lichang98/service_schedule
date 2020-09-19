/**
 * Monte Carlo Tree Search Algorithm
 * 
 */
#include "metrics.hpp"
#include <random>

// Monte Carlo Tree Node Structure
// The Node records the state of tasks and experts at certain time stamp
struct MCTreeNode
{
    int env_tm;
    bool is_leaf;
    bool is_terminate_stat;
    int num_sim;
    double sum_reward;
    MCTreeNode *parent;
    int num_finish_tasks;
    std::vector<utils::Task> task_status;
    std::vector<utils::Expert> expert_status;
    std::vector<MCTreeNode *> children_nodes;

    MCTreeNode() : env_tm(0), is_leaf(true), is_terminate_stat(false), num_sim(0),
                   sum_reward(0), num_finish_tasks(0), parent(nullptr) {}

    MCTreeNode &operator=(const MCTreeNode &node)
    {
        this->env_tm = node.env_tm;
        this->is_leaf = node.is_leaf;
        this->is_terminate_stat = node.is_terminate_stat;
        this->num_sim = 0;
        this->sum_reward = 0;
        this->parent = nullptr;
        this->num_finish_tasks = node.num_finish_tasks;
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
void expand(MCTreeNode *root, int num_sim = 10)
{
    // randomly select valid action to expand new nodes, simulated to terminal state
    // and backpropgate the rewards
    int env_tm = root->env_tm + 1;
    std::default_random_engine random_gen;
    std::uniform_int_distribution<int> dist(0, root->task_status.size() - 1);
    std::uniform_int_distribution<int> dist_action_with_wait(0, root->expert_status.size());
    std::uniform_int_distribution<int> dist_action_no_wait(0, root->expert_status.size() - 1);

    while (num_sim-- > 0)
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

int main(int argc, char const *argv[])
{
    std::vector<utils::Task> tasks = utils::load_work_order();
    std::vector<utils::Expert> experts = utils::load_expert_process_duras();
    int num_types = experts[0].num_types();

    return 0;
}
