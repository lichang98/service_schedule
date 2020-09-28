/**
 * This file contains utils for monte carlo method
 */
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace monte_utils
{
const static char WORK_ORDER[100] = "../data/SIC_round1_testA_20200909/work_order.csv";
const static char PROCESS_TM_MAT[100] = "../data/SIC_round1_testA_20200909/process_time_matrix.csv";
static char PRED_RESULT_PREFIX[100] = "../prediction_result/submit_";
size_t PROCESS_TM_FILE_LINE_MAXLEN = 1000;
const static int EXPERT_MAX_PARALLEL = 3;
const static int EXPERT_NOT_GOOD_TIME = 999999; // If expert not good at some type tasks, the processing time will be 999999
const static int TASK_MAX_MIGRATION = 5;
const static int NUM_TASK_TYPE = 107;

struct Task
{
    int task_id;
    int generate_tm;
    int type;
    int max_resp;

    // variables for recording and scoring
    int start_process_tm;
    int finish_tm;
    int assign_tm[TASK_MAX_MIGRATION];
    int each_stay_expert_id[TASK_MAX_MIGRATION];
    int curr_migrate_count;

    Task() : task_id(-1), generate_tm(-1), type(-1), max_resp(-1),
             start_process_tm(-1), finish_tm(-1), curr_migrate_count(0)
    {
        for (int i = 0; i < TASK_MAX_MIGRATION; ++i)
        {
            each_stay_expert_id[i] = -1;
            assign_tm[i] = -1;
        }
    }

    Task(int _task_id, int _generate_tm, int _type, int _max_resp) : task_id(_task_id),
                                                                     generate_tm(_generate_tm), type(_type), max_resp(_max_resp),
                                                                     start_process_tm(-1), finish_tm(-1), curr_migrate_count(0)
    {
        for (int i = 0; i < TASK_MAX_MIGRATION; ++i)
        {
            each_stay_expert_id[i] = -1;
            assign_tm[i] = -1;
        }
    }

    Task &operator=(const Task &task)
    {
        if (this != &task)
        {
            this->task_id = task.task_id;
            this->generate_tm = task.generate_tm;
            this->type = task.type;
            this->max_resp = task.max_resp;

            this->start_process_tm = task.start_process_tm;
            this->finish_tm = task.finish_tm;
            this->curr_migrate_count = task.curr_migrate_count;

            for (int i = 0; i < TASK_MAX_MIGRATION; ++i)
            {
                this->assign_tm[i] = task.assign_tm[i];
                this->each_stay_expert_id[i] = task.each_stay_expert_id[i];
            }
        }
        return *this;
    }
};

struct Expert
{
    int expert_id;
    int channels[EXPERT_MAX_PARALLEL]; // each channel processing a task
    int process_type_duras[NUM_TASK_TYPE];

    int num_idle_channel;
    int busy_sum; // the time long of processing takss from system begin to end

    Expert() : expert_id(-1), num_idle_channel(EXPERT_MAX_PARALLEL), busy_sum(0)
    {
        for (int i = 0; i < EXPERT_MAX_PARALLEL; ++i)
            channels[i] = -1;
        for (int i = 0; i < NUM_TASK_TYPE; ++i)
            process_type_duras[i] = EXPERT_NOT_GOOD_TIME;
    }

    Expert &operator=(const Expert &expt)
    {
        if (this != &expt)
        {
            this->expert_id = expt.expert_id;
            this->num_idle_channel = expt.num_idle_channel;
            this->busy_sum = expt.busy_sum;
            for (int i = 0; i < EXPERT_MAX_PARALLEL; ++i)
                this->channels[i] = expt.channels[i];
            for (int i = 0; i < NUM_TASK_TYPE; ++i)
                this->process_type_duras[i] = expt.process_type_duras[i];
        }
        return *this;
    }
};

std::vector<Task> load_tasks()
{
    std::vector<Task> tasks;
    FILE *fp = fopen(WORK_ORDER, "r");
    int task_id, generate_tm, type, max_resp;
    while (fscanf(fp, " %d,%d,%d,%d", &task_id, &generate_tm, &type, &max_resp) != EOF)
        tasks.emplace_back(Task(task_id, generate_tm, type - 1, max_resp));
    fclose(fp);
    return tasks;
}

std::vector<Expert> load_experts()
{
    std::vector<Expert> experts;
    FILE *fp = fopen(PROCESS_TM_MAT, "r");
    char *line = nullptr;
    getline(&line, &PROCESS_TM_FILE_LINE_MAXLEN, fp);
    int index = 1;
    while (getline(&line, &PROCESS_TM_FILE_LINE_MAXLEN, fp) != EOF)
    {
        Expert expt;
        expt.expert_id = index++;
        char *e = strtok(line, ",");
        e = strtok(nullptr, ",");
        int i = 0;
        while (e != nullptr)
        {
            expt.process_type_duras[i++] = atoi(e);
            e = strtok(nullptr, ",");
        }
        experts.emplace_back(expt);
    }
    fclose(fp);
    return experts;
}

} // namespace monte_utils
