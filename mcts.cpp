/**
 * Monte Carlo Tree Search Algorithm
 */

#include "monte_metrics.hpp"
#include "monte_utils.hpp"
#include <iostream>
#include <unistd.h>

#define RANDOM(low, high) ((int)(rand() % (high - low + 1) + low))

static std::vector<std::vector<int>> BEST_RESULT; // each elemment is task id, expert id, and assign time
static double BEST_SCORE = 0.0;
static const int FORCE_MIGRATE_MAX_EXEC = 1000; // if task has executed on a expert for more than the value, the task must be forced to migrate
static const int MAX_EXPAND_CHILD = 8;
static const int URGENT_THRESHOLD = 10;
static const int FAVOR_EPSILON = 30;
static const int MAX_SIMULATION_DEPTH = 10000;
static const int NUM_SIMULATION = MAX_EXPAND_CHILD * 8;
static const int MAX_ITER = 10000;

struct MCTNode
{
    int env_tm;
    int num_sim;
    double reward_sum;
    int num_finish_tasks;
    MCTNode *parent;
    std::vector<monte_utils::Task> tasks;
    std::vector<monte_utils::Expert> experts;
    MCTNode *child_nodes[MAX_EXPAND_CHILD];
    int child_node_count = 0;

    MCTNode() : env_tm(0), num_sim(0), reward_sum(0), num_finish_tasks(0), parent(nullptr), child_node_count(0)
    {
        for (int i = 0; i < MAX_EXPAND_CHILD; ++i)
            child_nodes[i] = nullptr;
    }

    ~MCTNode()
    {
        parent = nullptr;
        for (int i = 0; i < MAX_EXPAND_CHILD; ++i)
            child_nodes[i] = nullptr;
    }

    MCTNode &operator=(const MCTNode &node)
    {
        if (this != &node)
        {
            this->env_tm = node.env_tm;
            this->num_sim = node.num_sim;
            this->reward_sum = node.reward_sum;
            this->num_finish_tasks = node.num_finish_tasks;
            this->parent = node.parent;
            this->child_node_count = node.child_node_count;

            this->tasks.resize(node.tasks.size());
            for (int i = 0; i < node.tasks.size(); ++i)
                this->tasks[i] = node.tasks[i];
            this->experts.resize(node.experts.size());
            for (int i = 0; i < node.experts.size(); ++i)
                this->experts[i] = node.experts[i];
        }
        return *this;
    }

    void add_child(MCTNode *node)
    {
        this->child_nodes[this->child_node_count++] = node;
    }

    void remove_last_child()
    {
        this->child_nodes[--this->child_node_count] = nullptr;
    }

    void remove_free_last_child()
    {
        MCTNode *p = this->child_nodes[--this->child_node_count];
        this->child_nodes[this->child_node_count] = nullptr;
        delete p;
    }

    void clear_child_nodes()
    {
        for (int i = 0; i < this->child_node_count; ++i)
            this->child_nodes[i] = nullptr;
        this->child_node_count = 0;
    }

    void clear_free_child_nodes()
    {
        for (int i = 0; i < this->child_node_count; ++i)
        {
            delete this->child_nodes[i];
            this->child_nodes[i] = nullptr;
        }
        this->child_node_count = 0;
    }
};

MCTNode *init_root(std::vector<monte_utils::Task> &tasks, std::vector<monte_utils::Expert> &expts)
{
    MCTNode *root = new MCTNode();
    root->tasks.resize(tasks.size());
    for (int i = 0; i < root->tasks.size(); ++i)
        root->tasks[i] = tasks[i];
    root->experts.resize(expts.size());
    for (int i = 0; i < root->experts.size(); ++i)
        root->experts[i] = expts[i];
    return root;
}

std::vector<std::vector<int>> group_expert_by_type(std::vector<monte_utils::Expert> &experts, int num_types)
{
    std::vector<std::vector<int>> expert_groups(num_types);
    for (int i = 0; i < experts.size(); ++i)
    {
        for (int j = 0; j < num_types; ++j)
        {
            if (experts[i].process_type_duras[j] < monte_utils::EXPERT_NOT_GOOD_TIME)
                expert_groups[j].push_back(i);
        }
    }
    return expert_groups;
}

bool assign_task_to_expert(monte_utils::Task &task, monte_utils::Expert &expert, int task_idx, int expt_idx, int env_tm)
{
    if (expert.num_idle_channel <= 0)
        return false;
    if (task.start_process_tm == -1)
        task.start_process_tm = env_tm;
    task.assign_tm[task.curr_migrate_count] = env_tm;
    task.each_stay_expert_id[task.curr_migrate_count] = expt_idx;
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
    return true;
}

/**
 * Try to assign current task to suitable expert, if no suitable expert availble
 * then try no suitable expert
 */
bool try_assign_suit_expert(MCTNode *node, int env_tm, int selected_task_idx, std::vector<std::vector<int>> &expert_groups)
{
    int num_retry = 3;
    monte_utils::Task *task = &node->tasks[selected_task_idx];
    int rand_high = expert_groups[task->type].size() - 1;
    bool flag_assign = false;
    for (int i = 0; i < num_retry && !flag_assign; ++i)
    {
        int target_expert_idx = RANDOM(0, rand_high);
        target_expert_idx = expert_groups[task->type][target_expert_idx];
        flag_assign = assign_task_to_expert(node->tasks[selected_task_idx], node->experts[target_expert_idx],
                                            selected_task_idx, target_expert_idx, env_tm);
    }
    if (!flag_assign)
    {
        // not randomly fouond a suitable expert, traverse all suitable expert and then all other experts
        for (int i = 0; i < expert_groups[task->type].size() && !flag_assign; ++i)
        {
            flag_assign = assign_task_to_expert(node->tasks[selected_task_idx], node->experts[expert_groups[task->type][i]],
                                                selected_task_idx, expert_groups[task->type][i], env_tm);
        }
    }
    if (!flag_assign)
    {
        // try randomly select from not suitable experts
        for (int i = 0; i < expert_groups.size() && !flag_assign; ++i)
        {
            if (i == task->type)
                continue;
            for (int j = 0; j < num_retry && !flag_assign; ++j)
            {
                int idx = RANDOM(0, expert_groups[i].size() - 1);
                flag_assign = assign_task_to_expert(node->tasks[selected_task_idx], node->experts[expert_groups[i][idx]],
                                                    selected_task_idx, expert_groups[i][idx], env_tm);
            }
        }
    }
    if (!flag_assign)
    {
        // if still not found experts availables, traverse all experts
        for (int i = 0; i < expert_groups.size() && !flag_assign; ++i)
        {
            if (i == task->type)
                continue;
            for (int j = 0; j < expert_groups[i].size() && !flag_assign; ++j)
            {
                flag_assign = assign_task_to_expert(node->tasks[selected_task_idx], node->experts[expert_groups[i][j]],
                                                    selected_task_idx, expert_groups[i][j], env_tm);
            }
        }
    }
    task = nullptr;
    return flag_assign;
}

/**
 * This function is little bit different from `try_assign_suit_expert`
 * this function only try randomly select, and will not traverse if not assigned during random process
 */
bool try_assign_suit_or_wait(MCTNode *node, int env_tm, int selected_task_idx, std::vector<std::vector<int>> &expert_groups)
{
    int num_retry = 3;
    monte_utils::Task *task = &node->tasks[selected_task_idx];
    int rand_high = expert_groups[task->type].size() - 1;
    bool flag_assign = false;
    for (int i = 0; i < num_retry && !flag_assign; ++i)
    {
        int target_expert_idx = RANDOM(0, rand_high);
        target_expert_idx = expert_groups[task->type][target_expert_idx];
        flag_assign = assign_task_to_expert(node->tasks[selected_task_idx], node->experts[target_expert_idx],
                                            selected_task_idx, target_expert_idx, env_tm);
    }
    // if not randomly found suitable expert, traverse
    for (int i = 0; i < expert_groups[task->type].size() && !flag_assign; ++i)
        flag_assign = assign_task_to_expert(node->tasks[selected_task_idx], node->experts[expert_groups[task->type][i]],
                                            selected_task_idx, expert_groups[task->type][i], env_tm);
    task = nullptr;
}

void try_continue_exec_or_migrate(MCTNode *node, int env_tm, int selected_task_idx, std::vector<std::vector<int>> &expert_groups, bool force_next_suit)
{
    int favor_value = RANDOM(1, 100);
    monte_utils::Task *task = &node->tasks[selected_task_idx];
    int prev_expt_idx = task->each_stay_expert_id[task->curr_migrate_count - 1];
    bool is_migration = false;
    if (node->experts[prev_expt_idx].process_type_duras[task->type] == monte_utils::EXPERT_NOT_GOOD_TIME)
    {
        // favor migration
        if (favor_value > FAVOR_EPSILON)
        {
            // migration
            is_migration = true;
        }
    }
    else
    {
        // current expert is suitable, favor continuing executing
        if (favor_value <= FAVOR_EPSILON)
        {
            // migration
            is_migration = true;
        }
    }

    if (is_migration)
    {
        if (force_next_suit)
        {
            //  the next assigned expert must be suitable, if no avail, not migrate
            try_assign_suit_or_wait(node, env_tm, selected_task_idx, expert_groups);
        }
        else
        {
            // next assigned expert can be not suitable
            try_assign_suit_expert(node, env_tm, selected_task_idx, expert_groups);
        }
    }
    task = nullptr;
}

/**
 * check whether exist task finish and update record varialbes
 */
void update(MCTNode *node)
{
    for (int i = 0; i < node->experts.size(); ++i)
    {
        if (node->experts[i].num_idle_channel < monte_utils::EXPERT_MAX_PARALLEL)
            node->experts[i].busy_sum++;
    }
    for (int i = 0; i < node->tasks.size(); ++i)
    {
        if (node->tasks[i].curr_migrate_count == 0)
            continue;
        int task_due_time =
            node->experts[node->tasks[i].each_stay_expert_id[node->tasks[i].curr_migrate_count - 1]].process_type_duras[node->tasks[i].type] +
            node->tasks[i].assign_tm[node->tasks[i].curr_migrate_count - 1];
        if (task_due_time == node->env_tm)
        {
            // task finish
            node->num_finish_tasks++;
            node->tasks[i].finish_tm = node->env_tm;
            int expt_idx = node->tasks[i].each_stay_expert_id[node->tasks[i].curr_migrate_count - 1];
            for (int j = 0; j < monte_utils::EXPERT_MAX_PARALLEL; ++j)
            {
                if (node->experts[expt_idx].channels[j] == i)
                {
                    node->experts[expt_idx].channels[j] = -1;
                    node->experts[expt_idx].num_idle_channel++;
                    break;
                }
            }
        }
    }
}

/**
 * The expand operation during Monte Carlo Tree Search
 */
bool expand(MCTNode *root, std::vector<std::vector<int>> &expert_groups, int num_expand = MAX_EXPAND_CHILD)
{
    int env_tm = root->env_tm + 1;
    for (int ex = 0; ex < num_expand; ++ex)
    {
        MCTNode *child = new MCTNode();
        *child = *root;
        update(child);
        child->env_tm = env_tm;
        child->parent = root;
        root->add_child(child);
        for (int selected_task_idx = 0; selected_task_idx < child->tasks.size(); ++selected_task_idx)
        {
            if (child->tasks[selected_task_idx].finish_tm > 0 || child->tasks[selected_task_idx].generate_tm > env_tm)
                continue;
            // possible actions
            // if the task has not been assigned before, the task can choose wait or assign to an expert
            // the choice should depend on whther the task will soon timeout
            // if the task has been assigned, then it can choose continuing executing or migration
            // max migration restrict and whether the expert is suitable should be considered
            if (child->tasks[selected_task_idx].curr_migrate_count == 0)
            {
                // not assigned yet, wait or assign
                if (child->tasks[selected_task_idx].generate_tm + child->tasks[selected_task_idx].max_resp - env_tm < URGENT_THRESHOLD)
                {
                    // urgent, force assign if experts available, favor suitable expert
                    try_assign_suit_expert(child, env_tm, selected_task_idx, expert_groups);
                }
                else
                {
                    // not urgent, random choose, but favor assign
                    try_assign_suit_or_wait(child, env_tm, selected_task_idx, expert_groups);
                }
            }
            else
            {
                // have assigned, continuing executing or migrate
                if (child->tasks[selected_task_idx].curr_migrate_count + 1 == monte_utils::TASK_MAX_MIGRATION)
                {
                    // if current assigned expert is suitable, favor continue execution, not favor migration
                    // if not suitable, favor migration, but the last expert must be suitable
                    try_continue_exec_or_migrate(child, env_tm, selected_task_idx, expert_groups, true);
                }
                else if (child->tasks[selected_task_idx].curr_migrate_count + 1 < monte_utils::TASK_MAX_MIGRATION)
                {
                    // if current assigned expert is suitable, favor continue execution, not favor migration
                    // if not suitable, favor migration, but favor suit experts
                    try_continue_exec_or_migrate(child, env_tm, selected_task_idx, expert_groups, false);
                }
            }
        }
    }
}

/**
 * Extract solution from the node
 */
std::vector<std::vector<int>> extract_solution(MCTNode *node)
{
    std::vector<std::vector<int>> solution;
    for (int i = 0; i < node->tasks.size(); ++i)
    {
        for (int j = 0; j < node->tasks[i].curr_migrate_count; ++j)
        {
            solution.emplace_back(std::vector<int>({node->tasks[i].task_id,
                                                    node->tasks[i].each_stay_expert_id[j], node->tasks[i].assign_tm[j]}));
        }
    }
    return solution;
}

/**
 * The simulation procedure of Monte Carlo Tree Search start from node
 */
void simulate(MCTNode *node, std::vector<std::vector<int>> &expt_groups)
{
    MCTNode *p = node;
    int num_tasks = node->tasks.size();
    int simu_depth = 0;
    std::cout << "\tSimulation start..." << std::endl;
    while (p->num_finish_tasks < num_tasks)
    {
        expand(p, expt_groups, 1);
        MCTNode *q = p;
        p = p->child_nodes[0];
        if (q != node)
        {
            delete q;
            q = nullptr;
        }
        if (simu_depth++ == MAX_SIMULATION_DEPTH)
            break;
        if (simu_depth % 10 == 0)
            std::cout << "\t\tSimulation depth reach " << simu_depth << ", num tasks finish=" << p->num_finish_tasks << std::endl;
    }
    std::cout << "\tSimulation finish..." << std::endl;
    if (p->num_finish_tasks == num_tasks)
    {
        // finish all tasks, calculating reward
        node->reward_sum += monte_metrics::score(p->tasks, p->experts);
        node->num_sim++;
        std::cout << "\tSimulation reach finish state, reward accumulate=" << node->reward_sum << std::endl;
    }
    if (p->reward_sum / p->num_sim > BEST_SCORE)
    {
        BEST_RESULT = extract_solution(p);
        BEST_SCORE = p->reward_sum / p->num_sim;
        std::cout << "\tSimulation update best score=" << BEST_SCORE << std::endl;
    }
    if (p != node)
    {
        delete p;
        p = nullptr;
    }
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
void run_alg(MCTNode *root, std::vector<std::vector<int>> &expert_groups)
{
    std::vector<MCTNode *> leaf_nodes;
    leaf_nodes.push_back(root);
    std::cout << "Start Monte Carlo Tree Search..." << std::endl;
    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        sleep(1);
        std::cout << "Alg iter#" << iter << ":" << std::endl;
        MCTNode *best_leaf = leaf_nodes[0];
        for (int i = 0; i < leaf_nodes.size(); ++i)
        {
            if (leaf_nodes[i]->reward_sum / (leaf_nodes[i]->num_sim + __DBL_EPSILON__) >
                best_leaf->reward_sum / (best_leaf->num_sim + __DBL_EPSILON__))
                best_leaf = leaf_nodes[i];
        }
        for (int i = 0; i < leaf_nodes.size(); ++i)
        {
            if (leaf_nodes[i] == best_leaf)
            {
                leaf_nodes.erase(leaf_nodes.begin() + i);
                break;
            }
        }
        std::cout << "Expand best leaf node..." << std::endl;
        expand(best_leaf, expert_groups);
        for (int i = 0; i < best_leaf->child_node_count; ++i)
            leaf_nodes.push_back(best_leaf->child_nodes[i]);

        std::cout << "Simulate from expanded child nodes..." << std::endl;
        for (int i = 0; i < NUM_SIMULATION; ++i)
        {
            for (int j = 0; j < best_leaf->child_node_count; ++j)
                simulate(best_leaf->child_nodes[j], expert_groups);
        }
        // free best leaf node
        for (int i = 0; i < best_leaf->child_node_count; ++i)
            best_leaf->child_nodes[i]->parent = nullptr;
        delete best_leaf;
        best_leaf = nullptr;
    }
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));
    std::vector<monte_utils::Task> tasks = monte_utils::load_tasks();
    std::vector<monte_utils::Expert> experts = monte_utils::load_experts();
    std::vector<std::vector<int>> expert_groups = group_expert_by_type(experts, monte_utils::NUM_TASK_TYPE);
    MCTNode *root = init_root(tasks, experts);
    run_alg(root, expert_groups);
    return 0;
}
