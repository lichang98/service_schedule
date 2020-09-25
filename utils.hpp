/**
 *  This file is the util for file processing
 */
#pragma once

#include <cstdio>
#include <vector>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace utils
{
    const static char WORK_ORDER[100] = "../data/SIC_round1_testA_20200909/work_order.csv";
    const static char PROCESS_TM_MAT[100] = "../data/SIC_round1_testA_20200909/process_time_matrix.csv";
    static char PRED_RESULT_PREFIX[100] = "../prediction_result/submit_";
    size_t PROCESS_TM_FILE_LINE_MAXLEN = 1000;
    const static int EXPERT_MAX_PARALLEL = 3;
    const static int EXPERT_NOT_GOOD_TIME = 999999; // If expert not good at some type tasks, the processing time will be 999999
    const static int TASK_MAX_MIGRATION = 5;

    struct Expert;

    struct Task
    {
        int task_id;
        int tm_stamp;
        int type;
        int max_resp_tm;

        // variables for recording
        int start_process_tmpt;
        std::vector<int> each_stay_dura; // the stay duration on each via expert
        int finish_tmpt;
        // for monte carlo tree search
        int task_via_expert_idxs[TASK_MAX_MIGRATION];
        int task_assign_expert_tm[TASK_MAX_MIGRATION]; // The time when the task assigned to experts
        int task_stay_due_tm[TASK_MAX_MIGRATION];      // during the migration of tasks, the expected finish time on each expert
        int task_curr_via_count;

        Task() : task_id(-1), tm_stamp(-1), type(-1), max_resp_tm(-1), start_process_tmpt(-1), finish_tmpt(-1), task_curr_via_count(0)
        {
            for (int i = 0; i < TASK_MAX_MIGRATION; ++i)
            {
                task_via_expert_idxs[i] = -1;
                task_stay_due_tm[i] = -1;
                task_assign_expert_tm[i] = -1;
            }
        }

        Task(int _task_id, int _tm_stamp, int _type, int _max_resp_tm) : task_id(_task_id), tm_stamp(_tm_stamp),
                                                                         type(_type), max_resp_tm(_max_resp_tm), start_process_tmpt(-1),
                                                                         finish_tmpt(-1), task_curr_via_count(0)
        {
            for (int i = 0; i < TASK_MAX_MIGRATION; ++i)
            {
                task_via_expert_idxs[i] = -1;
                task_stay_due_tm[i] = -1;
                task_assign_expert_tm[i] = -1;
            }
        }

        ~Task()
        {
            this->each_stay_dura.clear();
        }

        Task &operator=(const Task &tsk)
        {
            if (this != &tsk)
            {
                this->task_id = tsk.task_id;
                this->tm_stamp = tsk.tm_stamp;
                this->type = tsk.type;
                this->max_resp_tm = tsk.max_resp_tm;
                this->start_process_tmpt = tsk.start_process_tmpt;
                this->each_stay_dura.clear();
                for (int val : tsk.each_stay_dura)
                    this->each_stay_dura.push_back(val);
                this->finish_tmpt = tsk.finish_tmpt;
                for (int i = 0; i < TASK_MAX_MIGRATION; ++i)
                {
                    this->task_via_expert_idxs[i] = tsk.task_via_expert_idxs[i];
                    this->task_assign_expert_tm[i] = tsk.task_assign_expert_tm[i];
                    this->task_stay_due_tm[i] = tsk.task_stay_due_tm[i];
                }
                this->task_curr_via_count = tsk.task_curr_via_count;
            }
            return *this;
        }

        friend std::ostream &operator<<(std::ostream &out, const Task &task)
        {
            out << "Task: ["
                << "task id=" << task.task_id << ", tm_stamp=" << task.tm_stamp << ", type=" << task.type << ", max_resp_tm=" << task.max_resp_tm
                << ", start_process_tmpt=" << task.start_process_tmpt << ", finish_tmpt=" << task.finish_tmpt
                << ", last_expert_dura="
                << "]";
            return out;
        }
    };

    struct Expert
    {
        int id;
        std::vector<int> process_dura;
        Task *process_tasks[EXPERT_MAX_PARALLEL];
        // for monte carlo
        int process_tasks_idxs[EXPERT_MAX_PARALLEL]; // if no task executed, the value is -1
        int process_remains[EXPERT_MAX_PARALLEL];
        int num_avail;
        // variables for record
        int busy_total_time; // the total time of expert processing coming tasks

        Expert()
        {
            std::fill(process_tasks, process_tasks + EXPERT_MAX_PARALLEL, nullptr);
            std::fill(process_remains, process_remains + EXPERT_MAX_PARALLEL, 0);
            std::fill(process_tasks_idxs, process_tasks_idxs + EXPERT_MAX_PARALLEL, -1);
            num_avail = EXPERT_MAX_PARALLEL;
            id = 0;
            busy_total_time = 0;
        }
        ~Expert()
        {
            for (int i = 0; i < EXPERT_MAX_PARALLEL; ++i)
            {
                if (process_tasks[i])
                {
                    delete process_tasks[i];
                    process_tasks[i] = nullptr;
                }
            }
        }

        Expert &operator=(const Expert &expt)
        {
            if (this != &expt)
            {
                this->id = expt.id;
                this->process_dura.resize(expt.process_dura.size());
                for (int i = 0; i < expt.process_dura.size(); ++i)
                    this->process_dura[i] = expt.process_dura[i];
                for (int i = 0; i < EXPERT_MAX_PARALLEL; ++i)
                {
                    this->process_tasks[i] = expt.process_tasks[i];
                    this->process_tasks_idxs[i] = expt.process_tasks_idxs[i];
                    this->process_remains[i] = expt.process_remains[i];
                }
                this->num_avail = expt.num_avail;
                this->busy_total_time = expt.busy_total_time;
            }
            return *this;
        }

        friend std::ostream &operator<<(std::ostream &out, const Expert &e)
        {
            out << "Expert id=" << e.id << ", time for " << e.process_dura.size() << " kinds of tasks, times are :[";
            for (int val : e.process_dura)
                out << val << ",";
            out << "]";
            return out;
        }

        int num_types()
        {
            return process_dura.size();
        }

        // Assign new task to current task
        bool monte_assign_task(int task_idx)
        {
            if (num_avail <= 0)
                return false;
            for (int i = 0; i < EXPERT_MAX_PARALLEL; ++i)
            {
                if (process_tasks_idxs[i] == -1)
                {
                    process_tasks_idxs[i] = task_idx;
                    num_avail--;
                    break;
                }
            }
            return true;
        }

        // A task leave the expert, it may be due to finishing or migration
        bool monte_release_task(int task_idx)
        {
            bool flag = false;
            for (int i = 0; i < EXPERT_MAX_PARALLEL; ++i)
            {
                if (process_tasks_idxs[i] == task_idx)
                {
                    process_tasks_idxs[i] = -1;
                    num_avail++;
                    flag = true;
                    break;
                }
            }
            return flag;
        }

        // Env time on slot forward
        void monte_one_tick()
        {
            if (num_avail < EXPERT_MAX_PARALLEL)
                busy_total_time++;
        }

        // The time elapsed one time slot, the expert process each task one time slot
        // Update the remains time
        std::vector<Task *> update(int tmpt)
        {
            std::vector<Task *> finish_tasks;
            bool is_working = false;
            for (int i = 0; i < EXPERT_MAX_PARALLEL; ++i)
            {
                // The recoding of last expert_dura and finish_tmpt may be different if using other strategy
                if (process_tasks[i])
                {
                    is_working = true;
                    Task *tsk = process_tasks[i];
                    process_remains[i]--;
                    if (process_remains[i] <= 0)
                    {
                        finish_tasks.push_back(tsk);
                        tsk->each_stay_dura.push_back(this->process_dura[tsk->type]);
                        tsk->finish_tmpt = tmpt;
                        process_tasks[i] = nullptr;
                        num_avail++;
                    }
                }
            }
            if (is_working)
                this->busy_total_time++;
            return finish_tasks;
        }

        bool assign_task(Task *task)
        {
            if (num_avail <= 0)
                return false;
            int task_type = task->type;
            int process_time = process_dura[task_type];
            for (int i = 0; i < EXPERT_MAX_PARALLEL; ++i)
            {
                if (!process_tasks[i])
                {
                    process_remains[i] = process_time;
                    process_tasks[i] = task;
                    break;
                }
            }
            num_avail--;
            return true;
        }
    };

    // Load work order content
    // Each line contains four integers
    // Task ID, Task generating time, type, max response time
    std::vector<Task> load_work_order()
    {
        std::vector<Task> tasks;
        FILE *fp = fopen(WORK_ORDER, "r");
        int task_id, tm, type, max_resp;
        while (fscanf(fp, " %d,%d,%d,%d", &task_id, &tm, &type, &max_resp) != EOF)
        {
            tasks.emplace_back(Task(task_id, tm, type - 1, max_resp)); // change type start from index 1 to 0
        }
        fclose(fp);
        return tasks;
    }

    std::vector<Expert> load_expert_process_duras()
    {
        std::vector<Expert> experts;
        FILE *fp = fopen(PROCESS_TM_MAT, "r");
        char *line = nullptr;
        getline(&line, &PROCESS_TM_FILE_LINE_MAXLEN, fp);
        int index = 1;
        while (getline(&line, &PROCESS_TM_FILE_LINE_MAXLEN, fp) != EOF)
        {
            Expert expert;
            expert.id = index++;
            char *e = strtok(line, ",");
            e = strtok(nullptr, ",");
            while (e != nullptr)
            {
                expert.process_dura.push_back(atoi(e));
                e = strtok(nullptr, ",");
            }
            experts.emplace_back(expert);
        }
        fclose(fp);
        return experts;
    }

    void save_result(char *result_filename, std::vector<std::vector<int>> &result)
    {
        FILE *fp = fopen(result_filename, "w+");
        for (int i = 0; i < result.size(); ++i)
            fprintf(fp, "%d,%d,%d\n", result[i][0], result[i][1], result[i][2]);
        fclose(fp);
    }

} // namespace utils
